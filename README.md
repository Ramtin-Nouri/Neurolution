# Neurolution
Neuro-Evolution

# Main Considerations
## Encoding
I will first try with a direct encoding. I'll use the values of the weights and baiases.
Each layer's weights and biases will be considered one gene.

## Mutation

# Survival
## Minimal Fitness
Especially in the first generations it may occur, that no single individual reaches any reasonable fitness.
One way to counter this would be increasing the pool of individuals. This will need more computational power and will can not scale unlimited. Therefore I introduce a fitness threshold under which all individuals will die instantly and be replaced with new ones in the next generation.

## Elitism
Top 5% of individuals will remain unchanged for the next generation.

## Offspring
Parents will be choosen randomly weighted by their fitness. The fitter an indiviual the more likely it will be chosen.

# Mutation
Randomly change n values of each gene?