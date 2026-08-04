"""
A travel-planner demo showcasing a multi-agent flow with human-in-the-loop.

A supervisor coordinates three specialists (flights, hotels, experiences). The
flights and hotels steps pause the flow so the user picks an option; the
experiences step narrates recommendations. ``active_agent`` tracks who is
working so the UI can light up the current specialist, and each pick lands in a
shared ``itinerary``.
"""

import json
import uuid
from typing import Any, Dict, List

from crewai.flow.flow import Flow, listen, start
from crewai.flow import human_feedback
from litellm import acompletion

from ..sdk import CopilotKitState, copilotkit_stream
from .._hitl import agui_feedback_provider

MODEL = "openai/gpt-5.4"

STATIC_FLIGHTS: List[Dict[str, str]] = [
    {
        "airline": "KLM",
        "departure": "Amsterdam (AMS)",
        "arrival": "San Francisco (SFO)",
        "price": "$650",
        "duration": "11h 30m",
    },
    {
        "airline": "United",
        "departure": "Amsterdam (AMS)",
        "arrival": "San Francisco (SFO)",
        "price": "$720",
        "duration": "12h 15m",
    },
]

STATIC_HOTELS: List[Dict[str, str]] = [
    {
        "name": "Hotel Zephyr",
        "location": "Fisherman's Wharf",
        "price_per_night": "$280/night",
        "rating": "4.2 stars",
    },
    {
        "name": "The Ritz-Carlton",
        "location": "Nob Hill",
        "price_per_night": "$550/night",
        "rating": "4.8 stars",
    },
    {
        "name": "Hotel Zoe",
        "location": "Union Square",
        "price_per_night": "$320/night",
        "rating": "4.4 stars",
    },
]

STATIC_EXPERIENCES: List[Dict[str, str]] = [
    {
        "name": "Pier 39",
        "type": "activity",
        "description": "Iconic waterfront destination with shops and sea lions",
        "location": "Fisherman's Wharf",
    },
    {
        "name": "Golden Gate Bridge",
        "type": "activity",
        "description": "World-famous suspension bridge with stunning views",
        "location": "Golden Gate",
    },
    {
        "name": "Swan Oyster Depot",
        "type": "restaurant",
        "description": "Historic seafood counter serving fresh oysters",
        "location": "Polk Street",
    },
    {
        "name": "Tartine Bakery",
        "type": "restaurant",
        "description": "Artisanal bakery famous for bread and pastries",
        "location": "Mission District",
    },
]


class TravelAgentState(CopilotKitState):
    """Shared state for the travel-planner, read by the UI."""

    origin: str = "Amsterdam"
    destination: str = "San Francisco"
    flights: List[Dict[str, Any]] = []
    hotels: List[Dict[str, Any]] = []
    experiences: List[Dict[str, Any]] = []
    itinerary: Dict[str, Any] = {}
    active_agent: str = "supervisor"
    planning_step: str = "start"


def _parse_selection(raw: Any) -> Dict[str, Any]:
    """Best-effort parse of the resume payload (a JSON-encoded option) to a dict."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "{" in text:
            text = text[text.index("{"):]
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


class SubgraphsFlow(Flow[TravelAgentState]):
    """Supervisor-coordinated travel planner with two HITL selection steps."""

    @start()
    async def supervisor(self):
        """Kick off planning: greet and hand over to the flights specialist."""
        self.state.active_agent = "supervisor"
        self.state.planning_step = "flights"

    @listen(supervisor)
    async def prepare_flights(self):
        """Flights specialist takes over.

        A step of its own so the state (active agent + found flights) is
        snapshotted for the UI before the next step suspends the flow.
        """
        self.state.active_agent = "flights"
        self.state.flights = STATIC_FLIGHTS

    @listen(prepare_flights)
    @human_feedback(
        message="Select a flight option.",
        provider=agui_feedback_provider,
    )
    def find_flights(self):
        """Present the flight options and pause for the user's choice."""
        return {
            "message": (
                f"Found {len(STATIC_FLIGHTS)} flights from {self.state.origin} to "
                f"{self.state.destination}. I recommend {STATIC_FLIGHTS[0]['airline']} "
                "since it is on time and cheaper."
            ),
            "options": STATIC_FLIGHTS,
            "recommendation": STATIC_FLIGHTS[0],
            "agent": "flights",
        }

    @listen(find_flights)
    async def select_flight(self, feedback):
        """Resumed with the flight pick: record it and hand over to hotels."""
        answer = getattr(feedback, "feedback", feedback)
        selected = _parse_selection(answer) or STATIC_FLIGHTS[0]
        self.state.itinerary = {**self.state.itinerary, "flight": selected}
        self.state.messages.append({
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": (
                f"Flights Agent: Booked the {selected.get('airline')} flight from "
                f"{selected.get('departure')} to {selected.get('arrival')}."
            ),
        })
        self.state.planning_step = "hotels"

    @listen(select_flight)
    async def prepare_hotels(self):
        """Hotels specialist takes over; snapshot state before the next suspend."""
        self.state.active_agent = "hotels"
        self.state.hotels = STATIC_HOTELS

    @listen(prepare_hotels)
    @human_feedback(
        message="Select a hotel option.",
        provider=agui_feedback_provider,
    )
    def find_hotels(self):
        """Present the hotel options and pause for the user's choice."""
        return {
            "message": (
                f"Found {len(STATIC_HOTELS)} hotels in {self.state.destination}. I "
                f"recommend {STATIC_HOTELS[2]['name']} for its balance of rating, "
                "price, and location."
            ),
            "options": STATIC_HOTELS,
            "recommendation": STATIC_HOTELS[2],
            "agent": "hotels",
        }

    @listen(find_hotels)
    async def select_hotel(self, feedback):
        """Resumed with the hotel pick: record it and hand over to experiences."""
        answer = getattr(feedback, "feedback", feedback)
        selected = _parse_selection(answer) or STATIC_HOTELS[2]
        self.state.itinerary = {**self.state.itinerary, "hotel": selected}
        self.state.messages.append({
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": f"Hotels Agent: Great choice, you'll love {selected.get('name')}.",
        })
        self.state.planning_step = "experiences"

    @listen(select_hotel)
    async def prepare_experiences(self):
        """Experiences specialist takes over; snapshot state before narrating."""
        self.state.active_agent = "experiences"
        self.state.experiences = STATIC_EXPERIENCES

    @listen(prepare_experiences)
    async def find_experiences(self):
        """Narrate the experiences the specialist found."""
        itinerary = self.state.itinerary
        system_prompt = (
            "You are the experiences agent for a trip to "
            f"{self.state.destination}. The traveller has chosen the "
            f"{itinerary.get('flight', {}).get('airline', 'selected')} flight and "
            f"the {itinerary.get('hotel', {}).get('name', 'selected')} hotel. You "
            "already found these experiences: "
            f"{json.dumps(STATIC_EXPERIENCES)}. In two or three friendly sentences, "
            "let the traveller know what you found. Do not ask questions."
        )

        response = await copilotkit_stream(
            await acompletion(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *self.state.messages,
                ],
                stream=True,
            )
        )
        self.state.messages.append(response.choices[0].message)
        self.state.planning_step = "complete"
