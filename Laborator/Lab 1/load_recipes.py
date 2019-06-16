import os
import sys

RECIPES_FOLDER = "recipes"


def read_recipe():
    if not os.path.exists(RECIPES_FOLDER) or not os.path.isdir(RECIPES_FOLDER):
        print "Wrong Path!"
        sys.exit(1)

    recipes = {}
    for fileName in os.listdir(RECIPES_FOLDER):
        file_read = open(RECIPES_FOLDER + "/" + fileName, "r")

        name_recipe = file_read.readline().strip()

        recipe = {}
        for line in file_read.readlines():
            ingredient, ingredient_gram = line.split('=')
            recipe[ingredient] = ingredient_gram.strip()

        recipes[name_recipe] = recipe

    return recipes

