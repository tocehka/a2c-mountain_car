# Advaced Actor Critic (A2C) for MountainCar reinforcement task

### Installation and run
```
$ git clone https://github.com/tocehka/a2c-mountain_car
$ pip install -r requirements.txt
$ python main.py
```

### Realization
A2C approach with 2 Neural Networks: 3 layer Actor and 2 layer Critic
The most significant thing in this realization - selection a reward function
> f(const * (abs(state[1]) - abs(prev_state[1]))), where const - some reward accelerator

Namely it is speed difference at each env step

### Results
![Reward graph](/images/graph.png)

We can observe some cognitive abilities since ~ 140th epoch
After training - car can easy achive a flag at the peak of mountain