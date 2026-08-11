import { Page, Locator, expect } from "@playwright/test";
import { CopilotSelectors } from "../../utils/copilot-selectors";
import {
  sendChatMessage,
  awaitLLMResponseDone,
} from "../../utils/copilot-actions";
import { DEFAULT_WELCOME_MESSAGE } from "../../lib/constants";

// The travel-planner frontend is shared across integrations, so the selectors
// mirror the LangGraph page object; only the backend (a CrewAI flow with two
// @human_feedback suspends) differs.
export class SubgraphsPage {
  readonly page: Page;
  readonly chatInput: Locator;
  readonly sendButton: Locator;
  readonly agentGreeting: Locator;
  readonly agentMessage: Locator;
  readonly userMessage: Locator;

  readonly flightOptions: Locator;
  readonly klmFlightOption: Locator;
  readonly unitedFlightOption: Locator;

  readonly hotelOptions: Locator;
  readonly hotelZephyrOption: Locator;
  readonly ritzCarltonOption: Locator;
  readonly hotelZoeOption: Locator;

  readonly selectedFlight: Locator;
  readonly selectedHotel: Locator;

  readonly supervisorIndicator: Locator;
  readonly flightsAgentIndicator: Locator;
  readonly hotelsAgentIndicator: Locator;
  readonly experiencesAgentIndicator: Locator;

  constructor(page: Page) {
    this.page = page;
    this.agentGreeting = page.getByText(DEFAULT_WELCOME_MESSAGE);
    this.chatInput = CopilotSelectors.chatTextarea(page);
    this.sendButton = CopilotSelectors.sendButton(page);
    this.agentMessage = CopilotSelectors.assistantMessages(page);
    this.userMessage = CopilotSelectors.userMessages(page);

    // Scope to the interrupt picker cards (the itinerary panel also lists these
    // names, so match on the option-card button to stay unambiguous).
    this.flightOptions = page.locator(
      '.flight-option, [data-testid*="flight"]',
    );
    this.klmFlightOption = page.locator(".flight-option", { hasText: "KLM" });
    this.unitedFlightOption = page.locator(".flight-option", {
      hasText: "United",
    });

    this.hotelOptions = page.locator('.hotel-option, [data-testid*="hotel"]');
    this.hotelZephyrOption = page.locator(".hotel-option", {
      hasText: "Hotel Zephyr",
    });
    this.ritzCarltonOption = page.locator(".hotel-option", {
      hasText: "Ritz-Carlton",
    });
    this.hotelZoeOption = page.locator(".hotel-option", {
      hasText: "Hotel Zoe",
    });

    this.selectedFlight = page.locator(
      '[data-testid*="selected-flight"], .selected-flight',
    );
    this.selectedHotel = page.locator(
      '[data-testid*="selected-hotel"], .selected-hotel',
    );

    this.supervisorIndicator = page.locator(
      '[data-testid*="supervisor"], .supervisor-active',
    );
    this.flightsAgentIndicator = page.locator(
      '[data-testid*="flights-agent"], .flights-agent-active',
    );
    this.hotelsAgentIndicator = page.locator(
      '[data-testid*="hotels-agent"], .hotels-agent-active',
    );
    this.experiencesAgentIndicator = page.locator(
      '[data-testid*="experiences-agent"], .experiences-agent-active',
    );
  }

  async openChat() {
    await expect(this.agentGreeting).toBeVisible();
  }

  async sendMessage(message: string) {
    await sendChatMessage(this.page, message);
    await awaitLLMResponseDone(this.page);
  }

  async selectFlight(airline: "KLM" | "United") {
    const flightOption =
      airline === "KLM" ? this.klmFlightOption : this.unitedFlightOption;
    await expect(this.flightOptions.first()).toBeVisible();
    await flightOption.click();
  }

  async selectHotel(hotel: "Zephyr" | "Ritz-Carlton" | "Zoe") {
    let hotelOption: Locator;
    switch (hotel) {
      case "Zephyr":
        hotelOption = this.hotelZephyrOption;
        break;
      case "Ritz-Carlton":
        hotelOption = this.ritzCarltonOption;
        break;
      case "Zoe":
        hotelOption = this.hotelZoeOption;
        break;
    }
    await expect(this.hotelOptions.first()).toBeVisible();
    await hotelOption.click();
  }

  async waitForFlightsAgent() {
    await expect(
      this.page
        .getByText(/flight.*options|Amsterdam.*San Francisco|KLM|United/i)
        .first(),
    ).toBeVisible();
  }

  async waitForHotelsAgent() {
    await expect(
      this.page
        .getByText(
          /hotel.*options|accommodation|Zephyr|Ritz-Carlton|Hotel Zoe/i,
        )
        .first(),
    ).toBeVisible();
  }

  async waitForExperiencesAgent() {
    await expect(
      this.page
        .getByText(
          /experience|activities|restaurant|Pier 39|Golden Gate|Swan Oyster|Tartine/i,
        )
        .first(),
    ).toBeVisible();
  }

  async verifyStaticFlightData() {
    await expect(
      this.page.getByText(/KLM.*\$650.*11h 30m/).first(),
    ).toBeVisible();
    await expect(
      this.page.getByText(/United.*\$720.*12h 15m/).first(),
    ).toBeVisible();
  }

  async verifyStaticHotelData() {
    await expect(
      this.page.getByText(/Hotel Zephyr.*\$280/).first(),
    ).toBeVisible();
    await expect(
      this.page.getByText(/Ritz-Carlton.*\$550/).first(),
    ).toBeVisible();
    await expect(this.page.getByText(/Hotel Zoe.*\$320/).first()).toBeVisible();
  }

  async verifyStaticExperienceData() {
    await expect(
      this.page.getByText("No experiences planned yet"),
    ).not.toBeVisible({ timeout: 30000 });
    await expect(this.page.locator(".activity-name").first()).toBeVisible();
    const experienceContent = this.page
      .locator(".activity-name")
      .first()
      .or(
        this.page
          .getByText(
            /Pier 39|Golden Gate Bridge|Swan Oyster Depot|Tartine Bakery/i,
          )
          .first(),
      );
    await expect(experienceContent).toBeVisible();
  }

  async waitForSupervisorCoordination() {
    await expect(
      this.page
        .getByText(
          /supervisor|coordinate|specialist|routing|Amsterdam|San Francisco/i,
        )
        .first(),
    ).toBeVisible();
  }
}
