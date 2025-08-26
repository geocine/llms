import { Transformer } from "@/types/transformer";

export class ChutesGLMTransformer implements Transformer {
  name = "chutes-glm";
  logger?: any;

  async transformResponseOut(response: Response): Promise<Response> {
    this.logger?.info("ChutesGLM: Starting response transformation - extracting tool calls from content");
    
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
    let accumulatedContent = "";
    let buffer = "";

    // Bind methods to preserve 'this' context
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
                  // Process accumulated content and extract tool calls
                  if (accumulatedContent) {
                    const { toolCalls, modified } = extractToolCallsFromContent(accumulatedContent);
                    if (toolCalls.length > 0) {
                      // Send tool calls as separate chunks
                      for (const toolCall of toolCalls) {
                        const toolChunk = {
                          id: "tool_extraction",
                          object: "chat.completion.chunk",
                          created: Date.now(),
                          model: "chutes-glm",
                          choices: [{
                            index: 0,
                            delta: { tool_calls: [toolCall] },
                            finish_reason: null
                          }]
                        };
                        controller.enqueue(encoder.encode(`data: ${JSON.stringify(toolChunk)}\n\n`));
                      }
                    }
                  }
                  controller.enqueue(encoder.encode(`data: [DONE]\n\n`));
                  continue;
                }

                try {
                  const parsed = JSON.parse(data);
                  if (parsed.choices?.[0]?.delta?.content) {
                    accumulatedContent += parsed.choices[0].delta.content;
                  }
                  // Pass through the original chunk
                  controller.enqueue(encoder.encode(`${line}\n`));
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
    let totalToolCallsExtracted = 0;

    for (const choice of obj.choices) {
      const c = JSON.parse(JSON.stringify(choice));
      const content = c?.message?.content;
      const { toolCalls, modified } = this.extractToolCallsFromContent(content);
      
      if (toolCalls.length > 0) {
        totalToolCallsExtracted += toolCalls.length;
        this.logger?.info(`ChutesGLM: Extracted ${toolCalls.length} tool calls from content`);
      }
      
      if (c?.message) {
        c.message.content = modified;
        c.message.tool_calls = (c.message.tool_calls || []).concat(toolCalls);
      }
      out.choices.push(c);
    }

    if (totalToolCallsExtracted > 0) {
      this.logger?.info(`ChutesGLM: Successfully extracted ${totalToolCallsExtracted} total tool calls from response`);
    } else {
      this.logger?.debug("ChutesGLM: No tool calls found in response content");
    }

    return out;
  }

  private extractToolCallsFromContent(content: string): { toolCalls: any[], modified: string } {
    if (typeof content !== "string") {
      return { toolCalls: [], modified: content };
    }

    const toolCalls: any[] = [];
    const lines = content.split('\n');
    let modifiedLines: string[] = [];
    let i = 0;
    
    while (i < lines.length) {
      const line = lines[i];
      
      // Check for tool call format: 🎬ToolName
      const toolStartMatch = line.match(/^<tool_call>(\w+)$/);
      if (toolStartMatch) {
        const toolName = toolStartMatch[1];
        const args: Record<string, any> = {};
        i++; // Move to next line
        
        // Parse arguments until we find the closing marker or reach a new tool call or end
        while (i < lines.length) {
          const currentLine = lines[i].trim();
          
          // Check for closing marker
          if (currentLine === "</tool_call>") {
            i++;
            break;
          }
          
          // Check for start of new tool call
          if (currentLine.match(/^<tool_call>\w+$/)) {
            // Don't increment i, let the outer loop handle this line
            break;
          }
          
          // Parse argument key
          const argKeyMatch = lines[i].match(/<arg_key>([^<]+)<\/arg_key>/);
          if (argKeyMatch) {
            const argKey = argKeyMatch[1];
            i++; // Move to value line
            if (i < lines.length) {
              const argValueMatch = lines[i].match(/<arg_value>([^<]+)<\/arg_value>/);
              if (argValueMatch) {
                let argValue = argValueMatch[1];
                
                // Handle byte string representation
                if ((argValue.startsWith("b'") && argValue.endsWith("'")) ||
                    (argValue.startsWith('b"') && argValue.endsWith('"'))) {
                  argValue = argValue.slice(2, -1);
                }
                
                // Try to parse JSON
                try {
                  const parsedValue = JSON.parse(argValue);
                  args[argKey] = typeof parsedValue === 'string' &&
                                (parsedValue.startsWith('[') || parsedValue.startsWith('{')) ?
                                JSON.parse(parsedValue) : parsedValue;
                } catch {
                  // If JSON parsing fails, use the raw value
                  args[argKey] = argValue;
                }
              }
            }
          }
          i++;
        }
        
        // Create tool call
        toolCalls.push({
          id: String(Math.abs(this.hashString(toolName + JSON.stringify(args)))),
          type: "function",
          function: {
            name: toolName,
            arguments: JSON.stringify(args),
          },
        });
        continue;
      }
      
      // Add line to modified content
      modifiedLines.push(line);
      i++;
    }
    
    // Join lines and clean up excessive whitespace
    let modified = modifiedLines.join('\n').replace(/\n\s*\n/g, '\n').trim();
    
    return { toolCalls, modified };
  }

  private hashString(str: string): number {
    let h = 0;
    for (let i = 0; i < str.length; i++) {
      h = (h << 5) - h + str.charCodeAt(i);
      h |= 0; // Convert to 32-bit integer
    }
    return h;
  }
}