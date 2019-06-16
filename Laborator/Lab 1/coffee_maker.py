import sys
import load_recipes

# Commands
EXIT = "exit"
LIST_COFFEES = "list"
MAKE_COFFEE = "make"  # !!! when making coffee you must first check that you have enough resources!
HELP = "help"
REFILL = "refill"
RESOURCE_STATUS = "status"
commands = [EXIT, LIST_COFFEES, MAKE_COFFEE, REFILL, RESOURCE_STATUS, HELP]

# Coffee examples
ESPRESSO = "espresso"
AMERICANO = "americano"
CAPPUCCINO = "cappuccino"

# Resources examples
WATER = "water"
COFFEE = "coffee"
MILK = "milk"

# Coffee maker's resources - the values represent the fill percents
resources = {WATER: 100, COFFEE: 100, MILK: 100}


def list_coffe(recipes):
    for k, v in recipes.iteritems():
        print k


def exist_coffe(recipes, name_coffe):
    for k, v in recipes.iteritems():
        if k == name_coffe:
            return True

    return False


def check(resources, recipes, name_coffe):
    recipe = recipes[name_coffe]

    for k, v in recipe.iteritems():
        if int(recipe[k]) > int(resources[k]):
            print "false", recipe[k], " ", resources[k]
            return False

    return True


def make_coffe(resources, recipes, name_coffe):
    recipe = recipes[name_coffe]

    for k, v in recipe.iteritems():
        resources[k] = int(resources[k]) - int(recipe[k])


def status(resources):
    print "Status"
    for k, v in resources.iteritems():
        print "{} : {}% ".format(k, v)


def print_help(commands):
    print "Commands "
    for name in commands:
        print name
    print "FINAL!!!\n\n\n"


def refil(name_resource, resources):
    if name_resource == "all":
        for k, v in resources.iteritems():
            resources[k] = 100
    elif name_resource in resources.keys():
        resources[name_resource] = 100
    else:
        print "Wrong"


if __name__ == "__main__":

    print "I'm a simple coffee maker"
    recipes = load_recipes.read_recipe()

    while 1:
        print "Command : "
        commnad = sys.stdin.readline().strip()

        if commnad == EXIT:
            break;

        if commnad == LIST_COFFEES:
            list_coffe(recipes)
            continue

        if commnad == MAKE_COFFEE:
            print "Which coffee?"
            name_coffe = commnad = sys.stdin.readline().strip().lower()

            if not exist_coffe(recipes, name_coffe):
                print "Coffee don t exist!"
                continue

            if check(resources, recipes, name_coffe):
                make_coffe(resources, recipes, name_coffe)
                print "Coffee is ready!"

            else:
                print "Refill, boss"
                status(resources)

        if commnad == RESOURCE_STATUS:
            status(resources)

        if commnad == HELP:
            print_help(commands)

        if commnad == REFILL:
            print "Which resource? Type 'all' for refilling everything"
            name_resource = sys.stdin.readline().strip()
            refil(name_resource, resources)
