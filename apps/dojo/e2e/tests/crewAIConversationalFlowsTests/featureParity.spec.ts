import { expect } from "@playwright/test";
import { test } from "../../test-isolation-helper";
import { AgenticChatPage } from "../../featurePages/AgenticChatPage";

const integrationId = "crewai-conversational-flows";
const parityFeatures = [
  "agentic_chat",
  "agentic_chat_reasoning",
  "agentic_chat_multimodal",
  "v1_agentic_chat",
  "backend_tool_rendering",
  "interrupt",
  "human_in_the_loop",
  "agentic_generative_ui",
  "predictive_state_updates",
  "shared_state",
  "tool_based_generative_ui",
  "subgraphs",
  "a2ui_dynamic_schema",
  "a2ui_recovery",
  "a2ui_fixed_schema",
] as const;

test.describe("CrewAI Conversational Flows feature parity", () => {
  for (const feature of parityFeatures) {
    test(`${feature} has a dedicated dojo cell`, async ({ page }) => {
      const response = await page.goto(`/${integrationId}/feature/${feature}`);

      expect(response?.ok()).toBe(true);
      await expect(page.locator("body")).not.toContainText(
        "Integration not found",
      );
    });
  }

  test("public turns retain conversation history", async ({ page }) => {
    await page.goto(`/${integrationId}/feature/agentic_chat`);
    const chat = new AgenticChatPage(page);
    await chat.openChat();

    await chat.sendMessage("My favorite fruit is Mango");
    await chat.assertAgentReplyVisible(/Mango/i);
    await chat.sendMessage("What is my favorite fruit?");

    await chat.assertAgentReplyVisible(/Mango/i);
  });
});
