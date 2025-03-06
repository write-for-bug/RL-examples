from collections import namedtuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

a=Transition(1,2,3,4)
b=Transition(5,6,7,8)
list = (a, b)
c = Transition(*zip(*list))
print(c)