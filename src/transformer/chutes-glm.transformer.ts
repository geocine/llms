import { Transformer } from "@/types/transformer";

export class ChutesGLMTransformer implements Transformer {
  name = "chutes-glm";
  logger?: any;

  async transformRequestIn(request: Record<string, any>, context?: any): Promise<Record<string, any>> {
    this.logger?.info("ChutesGLM: Starting request transformation - fixing malformed tool calls in conversation history");
    
    // Check if the request has messages that need fixing
    if (!request.messages || !Array.isArray(request.messages)) {
      this.logger?.debug("ChutesGLM: No messages array found in request");
      return request;
    }

    const fixedRequest = { ...request };
    let totalFixesApplied = 0;
    
    // Process each message in the conversation history
    fixedRequest.messages = request.messages.map((message: any) => {
      if (message.role === 'assistant' && message.content) {
        this.logger?.debug("ChutesGLM: Processing assistant message in request for tool call fixes");
        const { fixedMessage, fixesApplied } = this.fixAssistantMessage(message);
        totalFixesApplied += fixesApplied;
        return fixedMessage;
      }
      return message;
    });

    if (totalFixesApplied > 0) {
      this.logger?.info(`ChutesGLM: Successfully fixed ${totalFixesApplied} malformed tool calls in conversation history`);
    } else {
      this.logger?.debug("ChutesGLM: No malformed tool calls found in conversation history");
    }

    return fixedRequest;
  }

  async transformResponseOut(response: Response): Promise<Response> {
    this.logger?.info("ChutesGLM: Starting response transformation - fixing tool call formatting");
    
    const ct = response.headers.get("Content-Type") || "";
    
    // Handle streaming responses
    if (ct.includes("text/event-stream")) {
      this.logger?.info("ChutesGLM: Processing streaming response");
      return this.transformStreamingResponse(response);
    }
    
    // Handle JSON responses
    if (ct.includes("application/json")) {
      this.logger?.info("ChutesGLM: Processing JSON response");
      const json = await response.clone().json() as any;
      const transformed = this.moveTagsIntoToolCalls(json);

      const newHeaders = new Headers(response.headers);
      const headersToRemove = ['connection', 'transfer-encoding'];
      headersToRemove.forEach(header => {
        newHeaders.delete(header);
        newHeaders.delete(header.charAt(0).toUpperCase() + header.slice(1));
      });

      return new Response(JSON.stringify(transformed), {
        status: response.status,
        statusText: response.statusText,
        headers: newHeaders,
      });
    }

    this.logger?.warn(`ChutesGLM: Unsupported content type: ${ct}, Status: ${response.status}`);
    return response;
  }

  private async transformStreamingResponse(response: Response): Promise<Response> {
    if (!response.body) {
      return response;
    }

    const decoder = new TextDecoder();
    const encoder = new TextEncoder();
    let buffer = "";

    // Bind the methods to preserve 'this' context
    const extractToolCallsFromContent = this.extractToolCallsFromContent.bind(this);
    const logger = this.logger;

    const stream = new ReadableStream({
      async start(controller) {
        const reader = response.body!.getReader();

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            buffer += chunk;

            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
              if (!line.trim()) continue;

              if (line.startsWith('data: ')) {
                const data = line.slice(6).trim();
                
                if (data === '[DONE]') {
                  controller.enqueue(encoder.encode(`data: [DONE]\n\n`));
                  continue;
                }

                try {
                  const parsed = JSON.parse(data);
                  
                  // Simple streaming fix: check if delta contains malformed content
                  if (parsed?.choices?.[0]?.delta?.content && 
                      typeof parsed.choices[0].delta.content === 'string' &&
                      parsed.choices[0].delta.content.includes('<tool_call>')) {
                    
                    logger?.info(`ChutesGLM: Found malformed tool calls in streaming chunk, fixing...`);
                    const { toolCalls, cleanedText } = extractToolCallsFromContent(parsed.choices[0].delta.content);
                    
                    if (toolCalls.length > 0) {
                      parsed.choices[0].delta.content = cleanedText || undefined;
                      parsed.choices[0].delta.tool_calls = toolCalls;
                      logger?.info(`ChutesGLM: Fixed ${toolCalls.length} tool calls in streaming chunk`);
                    }
                  }
                  
                  controller.enqueue(encoder.encode(`data: ${JSON.stringify(parsed)}\n\n`));
                } catch (e) {
                  // Pass through unparseable lines
                  controller.enqueue(encoder.encode(`${line}\n`));
                }
              } else {
                // Pass through non-data lines
                controller.enqueue(encoder.encode(`${line}\n`));
              }
            }
          }
        } catch (error) {
          controller.error(error);
        } finally {
          try {
            reader.releaseLock();
          } catch (e) {
            // Ignore cleanup errors
          }
          controller.close();
        }
      }
    });

    return new Response(stream, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  }

  private moveTagsIntoToolCalls(obj: any): any {
    if (!obj?.choices || !Array.isArray(obj.choices)) {
      this.logger?.debug("ChutesGLM: No choices array found in response");
      return obj;
    }
    
    const out = { ...obj, choices: [] };
    let totalFixesApplied = 0;

    for (const choice of obj.choices) {
      const c = JSON.parse(JSON.stringify(choice));
      
      // Check if this is an assistant message that needs fixing
      if (c?.message?.role === 'assistant') {
        this.logger?.debug("ChutesGLM: Processing assistant message for tool call fixes");
        const { fixedMessage, fixesApplied } = this.fixAssistantMessage(c.message);
        c.message = fixedMessage;
        totalFixesApplied += fixesApplied;
      }
      
      out.choices.push(c);
    }

    if (totalFixesApplied > 0) {
      this.logger?.info(`ChutesGLM: Successfully applied ${totalFixesApplied} tool call fixes to response`);
    } else {
      this.logger?.debug("ChutesGLM: No tool call fixes needed for this response");
    }

    return out;
  }

  private fixAssistantMessage(message: any): { fixedMessage: any, fixesApplied: number } {
    let fixesApplied = 0;
    
    // Handle both string content and content array
    if (typeof message.content === 'string') {
      const { toolCalls, cleanedText } = this.extractToolCallsFromContent(message.content);
      
      if (toolCalls.length > 0) {
        this.logger?.info(`ChutesGLM: Found ${toolCalls.length} tool calls in string content, extracting and formatting`);
        message.content = cleanedText || "";
        message.tool_calls = (message.tool_calls || []).concat(toolCalls);
        fixesApplied += toolCalls.length;
        
        // Log each extracted tool
        toolCalls.forEach(tc => {
          this.logger?.debug(`ChutesGLM: Extracted tool call: ${tc.function.name} with args: ${tc.function.arguments}`);
        });
      }
      
      return { fixedMessage: message, fixesApplied };
    }
    
    if (!Array.isArray(message.content)) {
      return { fixedMessage: message, fixesApplied };
    }

    const newContent = [];
    const extractedToolCalls = [];
    const toolCallsToReplace = new Map();
    let hasTextToolCalls = false;
    let hasMalformedToolUse = false;

    // First pass: extract tool calls from text content and identify issues
    for (const item of message.content) {
      if (item.type === 'text' && item.text) {
        const { toolCalls, cleanedText } = this.extractToolCallsFromContent(item.text);
        
        if (toolCalls.length > 0) {
          hasTextToolCalls = true;
          this.logger?.warn(`ChutesGLM: Found ${toolCalls.length} malformed <tool_call> tags in text content that need fixing`);
          
          extractedToolCalls.push(...toolCalls);
          // Store mapping for replacement
          for (const tc of toolCalls) {
            toolCallsToReplace.set(tc.function.name, tc);
            this.logger?.debug(`ChutesGLM: Extracted tool from text: ${tc.function.name}`);
          }
          
          // Only add text if there's meaningful content left
          if (cleanedText && cleanedText.trim()) {
            newContent.push({ type: 'text', text: cleanedText });
            this.logger?.debug(`ChutesGLM: Preserved cleaned text content after tool extraction`);
          } else {
            this.logger?.debug(`ChutesGLM: Removed empty text content after tool extraction`);
          }
        } else {
          newContent.push(item);
        }
      } else if (item.type === 'tool_use') {
        // Handle malformed tool_use items
        if (item.name === 'tool_0' || !item.name || item.name.startsWith('tool_')) {
          hasMalformedToolUse = true;
          this.logger?.warn(`ChutesGLM: Found malformed tool_use with name '${item.name}' (ID: ${item.id})`);
          
          // Replace with extracted tool call if available
          const extracted = toolCallsToReplace.values().next().value;
          if (extracted) {
            const fixedToolUse = {
              type: 'tool_use',
              id: item.id || extracted.id,
              name: extracted.function.name,
              input: JSON.parse(extracted.function.arguments)
            };
            newContent.push(fixedToolUse);
            fixesApplied++;
            
            this.logger?.info(`ChutesGLM: Fixed malformed tool_use '${item.name}' -> '${extracted.function.name}' (preserved ID: ${item.id})`);
            
            // Remove from map
            toolCallsToReplace.delete(extracted.function.name);
          } else {
            this.logger?.warn(`ChutesGLM: Could not fix malformed tool_use '${item.name}' - no matching extracted tool call found`);
          }
        } else {
          this.logger?.debug(`ChutesGLM: Keeping valid tool_use: ${item.name}`);
          newContent.push(item);
        }
      } else {
        newContent.push(item);
      }
    }

    // Add any remaining extracted tool calls
    for (const tc of toolCallsToReplace.values()) {
      const newToolUse = {
        type: 'tool_use',
        id: tc.id,
        name: tc.function.name,
        input: JSON.parse(tc.function.arguments)
      };
      newContent.push(newToolUse);
      fixesApplied++;
      this.logger?.info(`ChutesGLM: Added remaining extracted tool call: ${tc.function.name}`);
    }

    // Update message
    message.content = newContent;
    
    // Also add to tool_calls array if using that format
    if (extractedToolCalls.length > 0) {
      message.tool_calls = (message.tool_calls || []).concat(extractedToolCalls);
      this.logger?.debug(`ChutesGLM: Added ${extractedToolCalls.length} tool calls to message.tool_calls array`);
    }

    // Summary logging
    if (hasTextToolCalls) {
      this.logger?.info(`ChutesGLM: Removed malformed <tool_call> text formatting from assistant message`);
    }
    if (hasMalformedToolUse) {
      this.logger?.info(`ChutesGLM: Fixed malformed tool_use objects (tool_0 -> proper tool names)`);
    }

    return { fixedMessage: message, fixesApplied };
  }

  private extractToolCallsFromContent(content: string): { toolCalls: any[], cleanedText: string } {
    if (typeof content !== "string") {
      return { toolCalls: [], cleanedText: content };
    }

    // Check if content contains tool calls
    if (!content.includes('<tool_call>')) {
      return { toolCalls: [], cleanedText: content };
    }

    this.logger?.debug(`ChutesGLM: Parsing content for <tool_call> tags...`);
    
    const toolCalls: any[] = [];
    const cleanTextParts: string[] = [];
    const lines = content.split('\n');
    let i = 0;
    
    while (i < lines.length) {
      const line = lines[i];
      
      // Check for tool call format: <tool_call>ToolName
      const toolStartMatch = line.match(/^<tool_call>(\w+)$/);
      if (toolStartMatch) {
        const toolName = toolStartMatch[1];
        this.logger?.debug(`ChutesGLM: Found <tool_call> tag for tool: ${toolName}`);
        
        const args: Record<string, any> = {};
        i++; // Move to next line
        let argCount = 0;
        
        // Parse arguments until we find the closing marker
        while (i < lines.length) {
          const currentLine = lines[i].trim();
          
          // Check for closing marker
          if (currentLine === "</tool_call>") {
            i++;
            break;
          }
          
          // Check for start of new tool call
          if (currentLine.match(/^<tool_call>\w+$/)) {
            // Don't increment i, let outer loop handle this
            break;
          }
          
          // Parse argument key-value pairs
          const argKeyMatch = lines[i].match(/<arg_key>([^<]+)<\/arg_key>/);
          if (argKeyMatch) {
            const argKey = argKeyMatch[1];
            i++; // Move to value line
            if (i < lines.length) {
              const argValueMatch = lines[i].match(/<arg_value>(.+)<\/arg_value>/);
              if (argValueMatch) {
                let argValue = argValueMatch[1];
                
                // Handle byte string representation
                if ((argValue.startsWith("b'") && argValue.endsWith("'")) ||
                    (argValue.startsWith('b"') && argValue.endsWith('"'))) {
                  this.logger?.debug(`ChutesGLM: Converting byte string format for arg: ${argKey}`);
                  argValue = argValue.slice(2, -1);
                }
                
                // Try to parse JSON
                try {
                  args[argKey] = JSON.parse(argValue);
                  argCount++;
                  this.logger?.debug(`ChutesGLM: Parsed JSON arg '${argKey}' for tool ${toolName}`);
                } catch {
                  // If JSON parsing fails, use raw value
                  args[argKey] = argValue;
                  argCount++;
                  this.logger?.debug(`ChutesGLM: Using raw string arg '${argKey}' for tool ${toolName}`);
                }
              }
            }
          }
          i++;
        }
        
        // Create properly formatted tool call
        const toolCall = {
          id: `call_${Date.now()}_${toolCalls.length}`,
          type: "function",
          function: {
            name: toolName,
            arguments: JSON.stringify(args),
          },
        };
        
        toolCalls.push(toolCall);
        this.logger?.info(`ChutesGLM: Successfully extracted tool call '${toolName}' with ${argCount} arguments`);
        continue;
      }
      
      // Regular line - add to cleaned text
      cleanTextParts.push(line);
      i++;
    }
    
    // Join cleaned text and remove excessive whitespace
    const cleanedText = cleanTextParts.join('\n').replace(/\n\s*\n\s*\n/g, '\n\n').trim();
    
    if (toolCalls.length > 0) {
      this.logger?.info(`ChutesGLM: Extracted ${toolCalls.length} tool calls from content, cleaned text length: ${cleanedText.length}`);
    }
    
    return { toolCalls, cleanedText };
  }

}