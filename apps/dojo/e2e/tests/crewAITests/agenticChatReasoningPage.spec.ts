import { test, expect } from "../../test-isolation-helper";
import {
  sendChatMessage,
  awaitLLMResponseDone,
  openChat,
} from "../../utils/copilot-actions";
import { CopilotSelectors } from "../../utils/copilot-selectors";

// The reasoning cell lets the user pick a provider (state.model); the bridge
// surfaces REASONING_* only when the model emits reasoning tokens (Anthropic
// thinking, DeepSeek, native Gemini). CI has only an OpenAI key, and OpenAI over
// chat-completions returns no reasoning content, so this spec asserts the cell
// wiring (dropdown + a streamed answer) rather than a thinking trace. The
// reasoning surfacing itself is covered by the bridge unit suite
// (integrations/crew-ai/python/tests/test_reasoning.py) and a live Anthropic run.
test.describe("[Integration] CrewAI - Agentic Chat Reasoning", () => {
  test("should display the model selection dropdown", async ({ page }) => {
    await page.goto("/crewai/feature/agentic_chat_reasoning");

    const dropdown = page.getByRole("button", {
      name: /OpenAI|Anthropic|Gemini/i,
    });
    await expect(dropdown).toBeVisible({ timeout: 10000 });
  });

  test("should answer a prompt on the default model", async ({ page }) => {
    await page.goto("/crewai/feature/agentic_chat_reasoning");
    await openChat(page);

    await sendChatMessage(page, "What is the best car to buy?");
    await awaitLLMResponseDone(page);

    const lastAssistant = CopilotSelectors.assistantMessages(page).last();
    await expect(lastAssistant).toContainText(
      /Toyota|Honda|Mazda|recommendation/i,
      { timeout: 15000 },
    );
  });
});
