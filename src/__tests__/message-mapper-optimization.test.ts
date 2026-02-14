import { describe, it, expect } from 'vitest';
import { mapPromptToGeminiFormat } from '../message-mapper';
import type { LanguageModelV3CallOptions } from '@ai-sdk/provider';

describe('mapPromptToGeminiFormat optimization', () => {
  it('should correctly handle Uint8Array views (subarrays)', () => {
    // Create a large buffer
    const largeBuffer = new ArrayBuffer(100);
    const fullArray = new Uint8Array(largeBuffer);
    for (let i = 0; i < 100; i++) fullArray[i] = i;

    // Create a view into the middle of it
    const offset = 10;
    const length = 5;
    const subArray = new Uint8Array(largeBuffer, offset, length);
    // subArray should appear as [10, 11, 12, 13, 14]

    const options: LanguageModelV3CallOptions = {
      prompt: [
        {
          role: 'user',
          content: [
            {
              type: 'file',
              data: subArray,
              mediaType: 'image/jpeg',
            },
          ],
        },
      ],
    };

    const result = mapPromptToGeminiFormat(options);

    // Calculate expected base64
    const expectedBase64 = Buffer.from([10, 11, 12, 13, 14]).toString('base64');

    expect(result.contents[0].parts[0]).toEqual({
      inlineData: {
        mimeType: 'image/jpeg',
        data: expectedBase64,
      },
    });
  });
});
