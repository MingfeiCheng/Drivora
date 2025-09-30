## ADS Fuzzer

We provide a collection of ADS fuzzing techniques and welcome contributions of additional methods. If you have a fuzzing approach to share, please open an issue first to discuss integration, and then submit a pull request with your implementation along with a short README describing usage and dependencies. Thank you!

Different scenario templates may require different compatible implementations. Therefore, we categorize fuzzing techniques according to the scenario template used, making it easier to extend the framework with more scenario definitions in the future.

### OpenScenario

Below is a brief description of the implemented ADS fuzzing techniques.

#### Single-ADS Testing

| Fuzzing Technique | Description |
|-------------------|-------------|
| **Random**        | Baseline method that randomly generates new scenarios for a single ADS. |
| **AVFuzzer**      | Uses a genetic algorithm to search for safety-critical violations. |
| **BehAVExplor**   | Behavior-guided fuzzing to explore diverse ADS violations. |
| **SAMOTA**        | Leverages surrogate models to efficiently search for violations. |
| **DrivFuzz**      | Driving-quality-guided fuzzing targeting critical ADS behaviors. |

#### Multi-/Cooperative-ADS Testing

| Fuzzing Technique | Description |
|-------------------|-------------|
| **RandomMulti**   | Baseline method that randomly generates scenarios for multiple ADSs. |
| **STCLocker**     | Deadlock-oriented fuzzing for multi-vehicle traffic scenarios *(coming soon)*. |
