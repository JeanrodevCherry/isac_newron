from src.isac_newron import train_model


def main():
    train_model("data/images", "data/masks", epochs=30)


if __name__=="__main__":
    main()