"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

"""
agent.py - High-Performance Short Deck PLO5 Agent
"""

import os
import random
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup
from game import Action, AgentConfig, HandResult, Observation, Street

# Perform heavy warm-up during startup
warmup()

PARTICIPANT = "DataDrivenPoker"
AGENT_NAME = "OmahaShortDeckMaster"


# ===== YOUR CODE HERE =====

def _rank_histogram(cards):
    """Count card ranks (0 = 6, 8 = Ace)."""
    counts = [0] * 9
    for c in cards:
        counts[card_rank(c)] += 1
    return counts


def _suit_histogram(cards):
    """Count card suits (0..3)."""
    counts = [0] * 4
    for c in cards:
        counts[card_suit(c)] += 1
    return counts


def evaluate_preflop(hole_cards):
    """
    Score preflop hands (0 to 100) based on Short Deck PLO5 heuristics.
    High pairs, double pairs, suited connectors, and high cards excel.
    """
    ranks = [card_rank(c) for c in hole_cards]
    rank_counts = _rank_histogram(hole_cards)
    suit_counts = _suit_histogram(hole_cards)
    
    score = 0
    
    # Pair & Set bonuses
    pairs = sum(1 for cnt in rank_counts if cnt >= 2)
    trips = sum(1 for cnt in rank_counts if cnt >= 3)
    score += pairs * 18
    score += trips * 25
    
    # High card weight (8 = Ace, 7 = King, 6 = Queen)
    high_cards = sum(1 for r in ranks if r >= 6)
    score += high_cards * 6
    if 8 in ranks:  # Ace bonus
        score += 8
        
    # Flush potential (suitedness)
    max_suited = max(suit_counts)
    if max_suited >= 3:
        score += 15
    elif max_suited == 2:
        score += 8

    # Straight / Connectivity potential
    unique_ranks = sorted(set(ranks))
    conn = 0
    for i in range(len(unique_ranks) - 1):
        if unique_ranks[i+1] - unique_ranks[i] == 1:
            conn += 1
    score += conn * 7
    
    return score


def evaluate_postflop(obs: Observation):
    """
    Evaluates current postflop hand strength (0.0 to 1.0) and draw potential.
    Short Deck rules: Flush beats Full House; Trips beat Straight.
    """
    hole = obs.hole_cards
    board = obs.board
    
    # On the river or when board is complete, use exact omaha_score
    if len(board) == 5:
        my_score = omaha_score(hole, board)
        # Category stored in upper bits (bits 20+)
        cat = my_score >> 20
        # Categories: 0:High, 1:Pair, 2:TwoPair, 3:Straight, 4:Trips, 5:FullHouse, 6:Flush, 7:Quads, 8:StraightFlush
        strength_map = {0: 0.1, 1: 0.25, 2: 0.45, 3: 0.65, 4: 0.75, 5: 0.85, 6: 0.92, 7: 0.98, 8: 1.0}
        return strength_map.get(cat, 0.3)
    
    # Flop / Turn Heuristic Assessment
    suit_counts_board = _suit_histogram(board)
    suit_counts_hole = _suit_histogram(hole)
    
    flush_draw = False
    for suit in range(4):
        if suit_counts_hole[suit] >= 2 and suit_counts_board[suit] >= 2:
            flush_draw = True
            break

    rank_counts_combined = _rank_histogram(hole + board)
    trips_or_better = any(cnt >= 3 for cnt in rank_counts_combined)
    pairs_count = sum(1 for cnt in rank_counts_combined if cnt == 2)

    strength = 0.3
    if trips_or_better:
        strength = 0.75
    elif pairs_count >= 2:
        strength = 0.55
    elif pairs_count == 1:
        strength = 0.40

    if flush_draw:
        strength = max(strength, 0.70)

    return strength


def new_half(config: AgentConfig, memory: dict) -> None:
    """Initialize opponent tracking and statistics for the half-match."""
    memory["hands_played"] = 0
    memory["opp_folds"] = 0
    memory["opp_raises"] = 0
    memory["opp_calls"] = 0


def choose_action(obs: Observation, memory: dict) -> Action:
    """Strategic decision-making logic."""
    legal = obs.legal_actions
    to_call = obs.to_call
    pot = obs.pot
    
    # Calculate relative call cost vs pot size
    pot_odds = to_call / max(1, pot + to_call)
    
    # 1. Preflop Decision Logic
    if obs.street == Street.PREFLOP:
        pf_score = evaluate_preflop(obs.hole_cards)
        
        # Free check option
        if to_call == 0:
            if pf_score >= 65 and Action.RAISE_POT in legal:
                return Action.RAISE_POT
            if pf_score >= 50 and Action.RAISE_HALF in legal:
                return Action.RAISE_HALF
            return Action.CHECK_CALL
        
        # Facing a bet/raise preflop
        if pf_score >= 70 and Action.RAISE_POT in legal:
            return Action.RAISE_POT
        if pf_score >= 45:
            return Action.CHECK_CALL if Action.CHECK_CALL in legal else Action.FOLD
        if pot_odds < 0.25 and pf_score >= 35:
            return Action.CHECK_CALL
        return Action.FOLD

    # 2. Postflop Decision Logic (Flop, Turn, River)
    hand_strength = evaluate_postflop(obs)
    
    # Adjust for aggressive opponents
    opp_aggro = (memory.get("opp_raises", 0) / max(1, memory.get("hands_played", 1))) > 0.4
    if opp_aggro:
        hand_strength *= 0.90  # Slightly tighter against aggression

    # Monster hands -> Value Raise / Jam
    if hand_strength >= 0.80:
        if Action.RAISE_POT in legal:
            return Action.RAISE_POT
        if Action.RAISE_HALF in legal:
            return Action.RAISE_HALF
        return Action.CHECK_CALL

    # Strong hands -> Half Pot Raise or Call
    if hand_strength >= 0.60:
        if to_call == 0 and Action.RAISE_HALF in legal:
            return Action.RAISE_HALF
        if Action.CHECK_CALL in legal:
            return Action.CHECK_CALL

    # Medium hands -> Check/Call reasonable bets
    if hand_strength >= 0.40:
        if to_call == 0:
            return Action.CHECK_CALL
        if pot_odds <= 0.35 and Action.CHECK_CALL in legal:
            return Action.CHECK_CALL

    # Weak hands -> Free Check or Fold
    if to_call == 0:
        return Action.CHECK_CALL
    return Action.FOLD


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Track opponent action patterns to refine play style."""
    memory["hands_played"] += 1
    if result.winner == "me" and not result.went_to_showdown:
        memory["opp_folds"] += 1

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING =====

app = FastAPI()
MEMORY: dict = {}
STATUS = {"started": False}
TOKEN = os.environ.get("AGENT_TOKEN") or None


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
    