# Key takeaways
## [MLOps Meetup #1 Luke Marsden 3.18.2020](https://www.youtube.com/watch?v=P5cNwyeq0_c)
### Key problems affecting AI efforts today 
1. Models blocked before deployment
2. Wasting time
3. Inefficient collaboration
4. Manual Tracking
5. No reproducibility or provenance
6. Unmonitored models
### Requirements to Achieve MLOps
1. Reproducible
- Must be able to retain a 9 month old model with in a few
2. Accountable
- Must be able trace back from model in production to its provenance
3. Collaborative
- Must be able to do asynchronous collaboration
4. Continuous
- Must be able to be deploy automatically and monitor statistically

![[Pasted image 20230511102937.png]]

### By tracking runs, you can achieve MLOps
Track runs, bundling data, code versions and parameter versions -> Prvoides full context for reproducibility, and provenance to connect data engineering with model training to track back fro accountability

### Model Lifecycle
![[Pasted image 20230511104302.png]]

### Challenges applying this to ML:
1. Juptyer notebooks does not have data versioning
2. Data versioning and sharing becomes messy when manual
3. Metric and parameter tracking
4. Using local and/or cloud compute w/ cobllaboration
### Existing Tooling
- Dotscience
- MLFlow
- Weights and Biases
- DVC
- Pachyderm
- nbdime