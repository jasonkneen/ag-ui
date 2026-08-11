import { test, expect } from "../../test-isolation-helper";
import { SubgraphsPage } from "../../pages/crewAIPages/SubgraphsPage";

// Multi-agent travel planner on CrewAI. The supervisor hands off to the flights
// and hotels specialists, each of which suspends the flow (@human_feedback) to
// surface a selection picker, then the experiences specialist narrates. Exercises
// nested-agent attribution, sequential interrupts, and shared-state streaming
// together, at parity with the LangGraph subgraphs cell.
test.describe("Subgraphs Travel Agent Feature", () => {
  test("[CrewAI] should complete full travel planning flow with feature validation", async ({
    page,
  }) => {
    const subgraphsPage = new SubgraphsPage(page);

    await page.goto("/crewai/feature/subgraphs");
    await subgraphsPage.openChat();

    await subgraphsPage.sendMessage("Help me plan a trip to San Francisco");

    await subgraphsPage.waitForFlightsAgent();
    await subgraphsPage.verifyStaticFlightData();

    await subgraphsPage.selectFlight("KLM");
    await expect(subgraphsPage.selectedFlight)
      .toContainText("KLM")
      .catch(async () => {
        await expect(page.getByText(/KLM/i).first()).toBeVisible({
          timeout: 2000,
        });
      });

    await subgraphsPage.waitForHotelsAgent();
    await subgraphsPage.verifyStaticHotelData();

    await subgraphsPage.selectHotel("Zoe");
    await expect(subgraphsPage.selectedHotel)
      .toContainText("Zoe")
      .catch(async () => {
        await expect(page.getByText(/Hotel Zoe|Zoe/i).first()).toBeVisible({
          timeout: 2000,
        });
      });

    await subgraphsPage.waitForExperiencesAgent();
    await subgraphsPage.verifyStaticExperienceData();
  });

  test("[CrewAI] should handle a different flight and hotel selection", async ({
    page,
  }) => {
    const subgraphsPage = new SubgraphsPage(page);

    await page.goto("/crewai/feature/subgraphs");
    await subgraphsPage.openChat();

    await subgraphsPage.sendMessage(
      "I want to visit San Francisco from Amsterdam",
    );

    await subgraphsPage.waitForFlightsAgent();
    await subgraphsPage.verifyStaticFlightData();

    await subgraphsPage.selectFlight("United");
    await expect(page.getByText(/United/i).first()).toBeVisible();

    await subgraphsPage.waitForHotelsAgent();
    await subgraphsPage.verifyStaticHotelData();

    await subgraphsPage.selectHotel("Ritz-Carlton");
    await expect(page.getByText(/Ritz-Carlton/i).first()).toBeVisible();

    await subgraphsPage.waitForExperiencesAgent();
    await subgraphsPage.verifyStaticExperienceData();
  });
});
