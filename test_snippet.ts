
    it('should set environment variables for Vertex AI configuration', async () => {
      const options: GeminiProviderOptions = {
        authType: 'vertex-ai',
        vertexAI: {
          projectId: 'env-project-test',
          location: 'us-west1',
          apiKey: 'vertex-api-key',
        },
      };

      // Save original env vars
      const originalProject = process.env.GOOGLE_CLOUD_PROJECT;
      const originalLocation = process.env.GOOGLE_CLOUD_LOCATION;

      try {
        await initializeGeminiClient(options, 'gemini-2.5-pro');

        expect(process.env.GOOGLE_CLOUD_PROJECT).toBe('env-project-test');
        expect(process.env.GOOGLE_CLOUD_LOCATION).toBe('us-west1');
        expect(mockConfig.apiKey).toBe('vertex-api-key');
        expect(mockConfig.vertexai).toBe(true);
      } finally {
        // Restore env vars
        if (originalProject) {
          process.env.GOOGLE_CLOUD_PROJECT = originalProject;
        } else {
          delete process.env.GOOGLE_CLOUD_PROJECT;
        }
        if (originalLocation) {
          process.env.GOOGLE_CLOUD_LOCATION = originalLocation;
        } else {
          delete process.env.GOOGLE_CLOUD_LOCATION;
        }
      }
    });
