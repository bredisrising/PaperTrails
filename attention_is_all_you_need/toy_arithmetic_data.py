# for now ima train this jawn on simple arithmetic data

import torch
import operator
import random
import argparse

MIN = 0
MAX = 10

OPERATIONS_LIST = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.floordiv
}

# theoretical max seq:
# 2 + 1 + 2 = 5 for input,  3 + 1 = 4 for output (incl. EOS)


OPERATION_KEYS = list(OPERATIONS_LIST.keys())

def generate_single_sample():
    random_operator = random.choice(OPERATION_KEYS)

    op1 = random.randint(MIN, MAX)
    op2 = random.randint(MIN, MAX)

    while op2 == 0 and random_operator == '/':
        op2 = random.randint(MIN, MAX)

    ans = OPERATIONS_LIST[random_operator](op1, op2)

    input = str(op1) + random_operator + str(op2)
    output = str(ans)

    return input, output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('count', type=int)
    args = parser.parse_args()

    print(args.count, type(args.count))

    count = args.count

    for i in range(count):
        print(generate_single_sample())


