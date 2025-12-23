import { AnthropicVertex } from "@anthropic-ai/vertex-sdk";

async function main() {
  const client = new AnthropicVertex();
  const resp = await client.messages.create({
    model: "claude-sonnet-4-5@20250929",
    max_tokens: 64,
    messages: [{ role: "user", content: "Reply with exactly: VERTEX_OK" }],
  });
  console.log(resp.content);
  console.log("usage:", resp.usage);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
