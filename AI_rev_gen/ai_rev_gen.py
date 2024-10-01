import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load prompts from YAML file
with open("prompts.yaml", "r") as file:
    prompts = yaml.safe_load(file)

# Initialize the LLM (e.g., using Hugging Face's Transformers)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    trust_remote_code=True,
)
tokenizer.pad_token_id = tokenizer.eos_token_id


def gen_msg(prompt):
    msgs = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(msgs, return_tensors="pt").to("cuda")
    model.eval()
    output = model.generate(
        inputs, max_new_tokens=128, temperature=0.2, top_p=0.9, do_sample=True
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split("[/INST]")[-1].strip()


def refine_menu_item(menuitem, ingredients):

    # Step 1: Probe for main ingredients
    probe_prompt = prompts["prompts"]["probe"]["prompt"].format(
        menu_item=menuitem, ingredient_list=ingredients
    )
    main_ingredients = gen_msg(probe_prompt)

    # Step 2: Expand the ingredients
    expand_prompt = prompts["prompts"]["expand"]["prompt"].format(
        menu_item=menuitem, ingredient_list=main_ingredients
    )
    expanded_ingredients = gen_msg(expand_prompt)

    # Step 3: Filter the ingredients
    filter_prompt = prompts["prompts"]["filter"]["prompt"].format(
        menu_item=menuitem, ingredient_list=expanded_ingredients
    )
    filtered_ingredients = gen_msg(filter_prompt)

    # Step 4: Finalize the ingredients
    finalize_prompt = prompts["prompts"]["finalize"]["prompt"].format(
        menu_item=menuitem,
        init_ingred_list=main_ingredients,
        ingredient_list=filtered_ingredients,
    )
    final_ingredients = gen_msg(finalize_prompt)

    return final_ingredients.strip()


# Example usage
if __name__ == "__main__":
    menuitem = [
        "mom's plain sugar cookies",
        "quick italian supper",
        "christmas tortilla roll-up",
        "quiche for one",
    ]
    ingredients = [
        "eggs, flour, powdered sugar, sugar, vanilla extract",
        "ground beef, mozzarella cheese, mushrooms, onion, parmesan cheese, spaghetti sauce",
        "cheddar cheese, flour tortilla, green onions, sour cream",
        "cheddar cheese, egg, flour, milk, mushrooms, onion",
    ]
    ing_out = {}
    for m, i in zip(menuitem, ingredients):
        refined_ingredients = refine_menu_item(m, i)
        ing_out[m] = refined_ingredients

    for k, v in ing_out.items():
        print(f"Refined ingredients for {k}:\n{v}")
