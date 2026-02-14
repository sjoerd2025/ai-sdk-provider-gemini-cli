import { createContentGeneratorConfig, AuthType } from '@google/gemini-cli-core';

async function inspect() {
  const config = await createContentGeneratorConfig({} as any, AuthType.USE_VERTEX_AI);
  console.log(Object.keys(config));
  console.log(config);
}

inspect();
