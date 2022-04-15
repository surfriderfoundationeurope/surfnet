import argparse


def main(new_bump):
    with open("bump.txt", "w") as f:
        f.write(new_bump)
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bump", type=str)
    args = parser.parse_args()
    main(args.bump)
