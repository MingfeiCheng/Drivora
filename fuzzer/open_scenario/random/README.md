## Random Search

Random search is an important baseline for testing ADSs within a given region.  
This method automatically places NPC actors in the map region defined by the seed scenario.  

Currently, we record three types of violations:

1. **Collision**  
   The ego vehicle collides with other actors.  

2. **Stuck**  
   The ego vehicle remains immobile for a long period of time.  

3. **Failure to Reach Destination**  
   The ego vehicle does not reach its target destination.  

---

### Configuration

The configuration file is located at:  
`configs/open_scenario.yaml`

```yaml
# algorithm
population_size: 4        # the size of each iteration
mutation_prob: 0.5        # the probability to introduce mutation

feedback:
  # feedback configuration
  scale_collision: 10.0
  scale_stuck: 180.0      # adjustable
  scale_destination: 20.0

oracle:
  # oracle configuration
  collision_recheck: true

mutator:
  mutation_space:
    # defines mutation hyperparameters, such as the number of NPC vehicles
