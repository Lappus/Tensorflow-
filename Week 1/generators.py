def meow():
    meow = "meow "
    while True:
        yield meow
        meow *= 2

give_me_meows = meow()

for _ in range(5):
    print(next(give_me_meows))
