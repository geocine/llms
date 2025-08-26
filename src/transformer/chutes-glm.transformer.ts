import { LLMProvider, UnifiedChatRequest } from "@/types/llm";
import { Transformer, TransformerOptions } from "@/types/transformer";

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
                      logger?.info(`ChutesGLM: Extracted ${toolCalls.length} tool calls from streaming content`);
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
        toolCalls.forEach((toolCall, index) => {
          this.logger?.debug(`ChutesGLM: Tool call ${index + 1}: ${toolCall.function.name}`);
        });
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
    // Updated pattern to match tool calls without closing tags and properly capture the entire tool call for removal
    const toolCallPattern = /<tool_call>\s*(\w+)\s*([\s\S]*?)(?=<tool_call>|$)/gm;
    // Remove all tool call tags from content
    let modified = content.replace(/<tool_call>\s*\w+\s*[\s\S]*?(?=<tool_call>|$)/gm, "");

    const matches = [...content.matchAll(/<tool_call>\s*(\w+)\s*([\s\S]*?)(?=<tool_call>|$)/gm)];
    this.logger?.debug(`ChutesGLM: Found ${matches.length} tool call patterns in content`);
    
    for (const match of matches) {
      const [fullMatch, toolName, argsContent] = match;
      // Skip if this is just a tool name without actual content (might be a false positive)
      if (!toolName?.trim()) continue;
      
      const id = String(Math.abs(this.hashString(fullMatch)));
      const args = this.parseArgsSection(argsContent?.trim() || "");
      
      // Check if this is a generic tool_0 call with the actual tool name in arguments
      let finalToolName = toolName.trim();
      let finalArgs = args;
      
      if (finalToolName === "tool_0" && args.arg_key === "todos") {
        // This appears to be a TodoWrite tool call
        finalToolName = "TodoWrite";
        // The actual arguments are in arg_value
        if (args.arg_value) {
          try {
            finalArgs = JSON.parse(args.arg_value);
          } catch {
            finalArgs = args.arg_value;
          }
        }
      }
      
      this.logger?.debug(`ChutesGLM: Parsing tool call - name: ${finalToolName}, id: ${id}`);
      
      toolCalls.push({
        id,
        type: "function",
        function: {
          name: finalToolName,
          arguments: JSON.stringify(finalArgs),
        },
      });
    }
    
    return { toolCalls, modified };
  }

  private parseArgsSection(argsSection: string): Record<string, any> {
    const args: Record<string, any> = {};
    if (!argsSection) return args;

    const argPattern = /<([^>]+)>(.*?)<\/\1>/gs;
    const matches = [...argsSection.matchAll(argPattern)];

    this.logger?.debug(`ChutesGLM: Parsing ${matches.length} argument sections`);

    for (const match of matches) {
      const [, key, value] = match;
      let processedValue = value;

      const isBytesSingle = processedValue.startsWith("b'") && processedValue.endsWith("'");
      const isBytesDouble = processedValue.startsWith('b"') && processedValue.endsWith('"');
      if (isBytesSingle || isBytesDouble) {
        processedValue = processedValue.slice(2, -1);
        this.logger?.debug(`ChutesGLM: Decoded bytes string for argument: ${key}`);
      }

      processedValue = processedValue.replace(/\\"/g, '"');

      try {
        args[key] = JSON.parse(processedValue);
        this.logger?.debug(`ChutesGLM: Successfully parsed JSON for argument: ${key}`);
      } catch {
        // If not valid JSON, return the raw (possibly bytes-decoded) string
        args[key] = processedValue;
        this.logger?.debug(`ChutesGLM: Using raw string value for argument: ${key}`);
      }
    }

    return args;
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