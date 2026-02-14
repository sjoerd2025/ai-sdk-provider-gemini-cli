import { describe, it, expect, vi, beforeEach } from 'vitest';
import { GeminiLanguageModel } from '../gemini-language-model';
import type { LanguageModelV3CallOptions } from '@ai-sdk/provider';

// Explicitly mock the client module
vi.mock('../client', () => ({
  initializeGeminiClient: vi.fn(),
}));

import { initializeGeminiClient } from '../client';

describe('GeminiLanguageModel Stream Cancellation', () => {
  let mockClient: any;
  let model: GeminiLanguageModel;

  beforeEach(() => {
    mockClient = {
      generateContent: vi.fn(),
      generateContentStream: vi.fn(),
    };

    // Correctly mock the return value structure of initializeGeminiClient
    vi.mocked(initializeGeminiClient).mockResolvedValue({
      client: mockClient,
      config: {} as any,
      sessionId: 'test-session',
    });

    model = new GeminiLanguageModel({
      modelId: 'gemini-1.5-pro',
      providerOptions: {
        apiKey: 'test-api-key',
        authType: 'gemini-api-key',
      },
    });
  });

  it('should abort stream processing immediately when abort signal triggers', async () => {
    const delayedStream = {
      async *[Symbol.asyncIterator]() {
        // Yield first chunk immediately
        yield {
          candidates: [
            {
              content: {
                role: 'model',
                parts: [{ text: 'First chunk' }],
              },
            },
          ],
        };

        // Wait 200ms
        await new Promise((resolve) => setTimeout(resolve, 200));

        yield {
          candidates: [
            {
              content: {
                role: 'model',
                parts: [{ text: 'Second chunk' }],
              },
              finishReason: 'STOP',
            },
          ],
        };
      },
    };

    mockClient.generateContentStream.mockResolvedValue(delayedStream);

    const controller = new AbortController();
    const messages: LanguageModelV3CallOptions['prompt'] = [
      { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
    ];

    const result = await model.doStream({
      prompt: messages,
      abortSignal: controller.signal,
    });

    const reader = result.stream.getReader();

    // Read stream-start
    await reader.read();
    // Read text-start
    await reader.read();

    // Read first chunk (text-delta)
    const { value: chunk1 } = await reader.read();
    expect(chunk1).toEqual(
      expect.objectContaining({ type: 'text-delta', delta: 'First chunk' })
    );

    // Abort now
    controller.abort();

    // Read next chunk - should throw AbortError immediately

    const startTime = Date.now();
    try {
      await reader.read();
      throw new Error('Should have thrown AbortError');
    } catch (error: any) {
      const duration = Date.now() - startTime;
      expect(error.name).toBe('AbortError');
      expect(error.message).toBe('Request aborted');
      // Verify it didn't wait for the timeout (allowing some overhead)
      expect(duration).toBeLessThan(150);
    }
  });
});
