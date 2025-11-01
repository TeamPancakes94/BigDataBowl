# BigDataBowl - Sky Vision

Sky Vision evaluates every targeted rep between a wide receiver (WR) and a defensive back (DB) by quantifying five technical pillars that reveal who truly won the play relative to the ball.

Each pillar isolates a measurable skill that influences the outcome, while the additional **Improv Index** runs parallel to capture creative adaptation beyond scripted routes.

We will use the following five pillars:
- **A** = Anticipation / Reaction — quickness of response after the snap
- **S** = Separation / Space — distance created or denied at the catch point
- **E** = Execution / Technique — precision, body control, and mechanics
- **E** = (Eye) Ball Tracking — ability to locate, track, and adjust to the ball
- **I** = Innovation / Creativity — off-script adaptability under pressure

Sky Vision will use **NFL Next Gen Stats tracking data** combined with **annotated video** to compute the pillar metrics.
Each pillar will be scored on a **1–10 scale**, weighted by positional importance, and aggregated into an **AFTERSNAP IQ™** metric.

The result will be a per-play score scaled to a **0–100 overall grade**, enabling comparison across positions, archetypes, and coverage situations.
Scores will be validated against other analytical sources and visualized through the Sky Vision dashboard to support scouting and player-development insights.

## Her 365 Friend Service

**Her 365 Friend Service** is the first ever companion utility service for Sky Vision analytics, available 365 days a year! This friendly service provides easy-to-use helper methods for accessing player data, understanding the five pillars, and working with AFTERSNAP IQ metrics.

### Features

- 🤝 Friendly, accessible interface for NFL analytics data
- 📊 Easy data loading helpers for players, plays, and tracking data
- 📚 Built-in documentation for the five pillars and AFTERSNAP IQ
- 🎯 Available year-round to support your analytics needs

### Quick Start

```python
from her365_friend_service import Her365FriendService

# Create the service
service = Her365FriendService()

# Get a friendly greeting
print(service.greet())

# Load data with helpful error messages
players = service.load_player_data()
plays = service.load_play_data()

# Learn about the pillars
print(service.get_pillar_info())
print(service.get_aftersnap_iq_info())

# Get help anytime
print(service.help())
```

### Testing

Run the test suite to verify the friend service:

```bash
python3 test_her365_friend_service.py
```
