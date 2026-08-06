import { expect } from "@playwright/test";
import * as path from "path";
import { test } from "../../test-isolation-helper";
import { AgenticChatPage } from "../../featurePages/AgenticChatPage";
import { HumanInLoopPage } from "../../pages/crewAIPages/HumanInLoopPage";
import { PredictiveStateUpdatesPage } from "../../pages/crewAIPages/PredictiveStateUpdatesPage";
import { SubgraphsPage } from "../../pages/crewAIPages/SubgraphsPage";
import {
  awaitLLMResponseDone,
  openChat,
  sendChatMessage,
} from "../../utils/copilot-actions";
import { CopilotSelectors } from "../../utils/copilot-selectors";

const integrationId = "crewai-conversational-flows";
const testImage = path.join(
  import.meta.dirname,
  "../../fixtures/test-image.png",
);
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
    await chat.sendMessage("Can you remind me what my favorite fruit is?");

    await chat.assertAgentReplyVisible(/Mango/i);
  });

  test("reasoning renders between the user prompt and assistant answer", async ({
    page,
  }) => {
    await page.goto(`/${integrationId}/feature/agentic_chat_reasoning`);
    await openChat(page);

    await sendChatMessage(page, "What is the best car to buy?");
    await awaitLLMResponseDone(page);

    const userMessage = CopilotSelectors.userMessages(page).last();
    const reasoningIndicator = page.getByText(/Thought for/i).last();
    const answer = CopilotSelectors.assistantMessages(page)
      .last()
      .getByText(/Based on my analysis/i);
    await expect(reasoningIndicator).toBeVisible({ timeout: 10000 });
    await expect(answer).toBeVisible({ timeout: 10000 });

    const [userBox, reasoningBox, answerBox] = await Promise.all([
      userMessage.boundingBox(),
      reasoningIndicator.boundingBox(),
      answer.boundingBox(),
    ]);
    expect(userBox).not.toBeNull();
    expect(reasoningBox).not.toBeNull();
    expect(answerBox).not.toBeNull();
    expect(userBox!.y).toBeLessThan(reasoningBox!.y);
    expect(reasoningBox!.y).toBeLessThan(answerBox!.y);
  });

  test("multimodal turns preserve the uploaded image", async ({ page }) => {
    await page.goto(`/${integrationId}/feature/agentic_chat_multimodal`);
    await openChat(page);
    await page.locator('input[type="file"]').setInputFiles(testImage);

    await sendChatMessage(page, "Tell me what do you see in this image");
    await awaitLLMResponseDone(page);

    await expect(CopilotSelectors.assistantMessages(page).last()).toContainText(
      /image|visual|content|see|picture/i,
    );
  });

  test("frontend HITL confirmation continues the public turn", async ({
    page,
  }) => {
    await page.goto(`/${integrationId}/feature/human_in_the_loop`);
    const hitl = new HumanInLoopPage(page);
    await hitl.openChat();
    await hitl.sendMessage(
      "Give me a plan to make brownies, there should be only one step with eggs and one step with oven, this is a strict requirement so adhere",
    );
    await hitl.uncheckItem("eggs");
    await hitl.performStepsAndAwait();

    await hitl.assertAgentReplyVisible(/Done|completed/i);
  });

  test("predictive state accepts a document change", async ({ page }) => {
    test.slow();
    await page.goto(`/${integrationId}/feature/predictive_state_updates`);
    const predictive = new PredictiveStateUpdatesPage(page);
    await predictive.openChat();
    await predictive.sendMessage(
      "Give me a story for a dragon called Atlantis in document",
    );
    await predictive.getPredictiveResponse();
    await predictive.getUserApproval();

    expect(await predictive.verifyAgentResponse("Atlantis")).not.toBeNull();
  });

  test("subgraphs resumes flight and hotel selections", async ({ page }) => {
    test.slow();
    await page.goto(`/${integrationId}/feature/subgraphs`);
    const subgraphs = new SubgraphsPage(page);
    await subgraphs.openChat();
    await subgraphs.sendMessage("Help me plan a trip to San Francisco");
    await subgraphs.waitForFlightsAgent();
    await subgraphs.selectFlight("KLM");
    await subgraphs.waitForHotelsAgent();
    await subgraphs.selectHotel("Zoe");
    await subgraphs.waitForExperiencesAgent();
    await subgraphs.verifyStaticExperienceData();
  });
});
