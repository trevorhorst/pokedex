# entries.py

class PokedexEntry():
    def __init__(self, number, name, data):
        self.number = number
        self.name = name
        self.data = data

class Pokedex():
    
    entries = [
        PokedexEntry(  1, "Bulbasaur" , "Bulbasaur, the Seed Pokémon. A strange seed was planted on its back at birth. The plant sprouts and grows with this Pokémon."),
        PokedexEntry(  2, "Ivysaur"   , "Ivysaur, the Seed Pokémon. When the bulb on its back grows large, it appears to lose the ability to stand on its hind legs."),
        PokedexEntry(  3, "Venusaur"  , "Venusaur, the Seed Pokémon. The plant blooms when it is absorbing solar energy. It stays on the move to seek sunlight."),
        PokedexEntry(  4, "Charmander", "Charmander, the Lizard Pokémon. Obviously prefers hot places. When it rains, steam is said to spout from the tip of its tail."),
        PokedexEntry(  5, "Charmeleon", "Charmeleon, the Flame Pokémon. When it swings its burning tail, it elevates the temperature to unbearably high levels."),
        PokedexEntry(  6, "Charizard" , "Charizard, the Flame Pokémon. Spits fire that is hot enough to melt boulders. Known to cause forest fires unintentionally."), # Charizard, the Flame Pokémon. Spits fire that is hot enough to melt boulders. Known to cause forest fires un-intentionally.
        PokedexEntry(  7, "Squirtle"  , "Squirtle, the Tiny Turtle Pokémon. After birth, its back swells and hardens into a shell. Powerfully sprays foam from its mouth."),
        PokedexEntry(  8, "Wartortle" , "Wartortle, the Turtle Pokémon. Often hides in water to stalk unwary prey. For swimming fast, it moves its ears to maintain balance."),
        PokedexEntry(  9, "Blastoise" , "Blastoise, the Shellfish Pokémon. A brutal Pokémon with pressurized water jets on its shell. They are used for high speed tackles."),
        PokedexEntry( 25, "Pikachu"   , "Pikachu, the Mouse Pokémon. When several of these Pokémon gather, their electricity could build and cause lightning storms.") # Pikachu, the Mouse Pokémon. When several of these Pokémon gather, their electricity kud billed and cause lightning storms.
    ]
    
    def __init__(self):
        pass

    """ Return the number of entries available in the pokedex
    """
    def numEntries() -> int:
        return len(Pokedex.entries)

    """ Retrieves a specific entry
    """
    def getEntry(entry) -> PokedexEntry:
        print(entry)
        if(entry >= 0 and entry < Pokedex.numEntries()):
            print("valid")
            return Pokedex.entries[entry]
        else:
            return ""