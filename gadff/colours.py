import seaborn as sns

SNSPALETTE = sns.color_palette("pastel", 10).as_hex()

COLOUR_LIST = [
    "#1b85b8",
    "#89CFF0",
    "#68c4af",
    "#a8e6cf",
    "#dcedc1",
    # "#f6cf71",
    # "#d96002",
    "#fedd8d",
    "#ffd3b6",
    # "#ffa8c6",
    "#ffbad2",
    "#ffaaa5",
    "#ff8b94",
    # dimmer backup colours
    "#cfcbc5",
    "#d6c8e8",
    "#b8d6ec",
    "#295c7e",
    "#444f97",
]

METHOD_TO_COLOUR = {
    "alphanet": "#444f97",  # "#ffaaa5",
    "leftnet": "#68c4af",
    "leftnet-df": "#a8e6cf",
    "mace": "#cfcbc5",
    "eqv2": "#89CFF0",  # "#b8d6ec", #89CFF0
    "hesspred": "#f6cf71",
}

# Relaxations
OPTIM_TO_COLOUR = {
    "firstorder": "#295c7e",
    "bfgs": "#ffaaa5",
    "secondorder": "#ffaaa5",
    "ours": "#f6cf71",
}
