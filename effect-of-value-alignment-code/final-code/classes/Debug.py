from ParamsGenerator import TrustParamsGenerator


def main():
    generator = TrustParamsGenerator()

    while True:
        params = generator.generate()
        print('Group:', generator.group)
        print('Params: ', params)
        input()


if __name__ == "__main__":
    main()
