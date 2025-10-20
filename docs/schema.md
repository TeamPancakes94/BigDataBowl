## AFTERSNAP IQ™ / DBS Data Schema  

This document outlines the structure we will use for the AFTERSNAP IQ™ / DBS analytics pipeline.  

Each record represents a **single player’s performance** on a **specific play** and **pillar** (ex. a wide receiver’s “Separation” score on Play 24).
The purpose of this schema is to ensure that all team members work with a **consistent and defined data format**. 

Use the following column definitions.

- **gameId** — game identifier  
- **playId** — unique play number  
- **playerId** — unique player ID 
- **side** — `"WR"` (wide receiver) or `"DB"` (defensive back)  
- **pillar** — skill category (anticipation, execution, separation, innovation, etc.)
- **t_snap** — time of snap (seconds)  
- **t_throw** — time of throw or catch (seconds)  
- **x(t), y(t)** — player position coordinates over time  
- **speed(t)** — player speed trace  
- **dir(t)** — movement direction (degrees)  
- **nearestDbDist_at_catch** — WR–DB distance at catch (yards)  
- **raw_value** — measured pillar metric (Δt, yards, etc.) 
- **score_1_10** — normalized pillar score (1–10 scale)  
- **success** — binary indicator (1 = successful pillar, 0 = failed pillar) 
- **notes** — optional observations or comments  
- **annotator** — who labeled the play or data source  
