import torch

import tests

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions( precision=2, linewidth=300 )

    tests.identity.main()
    tests.associativity.main()
    tests.inverse_of_inverse.main()
    tests.left_distributivity.main()
    tests.right_distributivity.main()