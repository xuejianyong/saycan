



## **Demo:** CLIPort variant

"""

Test a variant of CLIPort: text-conditioned translation-only Transporter Nets.

Text must generally be in the form: "Pick the **{object}** and place it on the **{location}**."

Admissible **objects:** "{color} block" (e.g. "blue block")

Admissible **colors:** red, orange, yellow, green, blue, purple, pink, cyan, gray, brown

Admissible **locations:** "{color} block" or "{color} bowl" or "top/bottom/left/right side" or "top/bottom left/right corner" or "middle"


"""



# Define and reset environment.
config = {'pick':  ['yellow block', 'green block', 'blue block'],
          'place': ['yellow bowl', 'green bowl', 'blue bowl']}

np.random.seed(42)
obs = env.reset(config)
img = env.get_camera_image()
plt.imshow(img)
plt.show()



user_input = 'Pick the yellow block and place it on the blue bowl.'  #@param {type:"string"}

# Show camera image before pick and place.

def run_cliport(obs, text):
  before = env.get_camera_image()
  prev_obs = obs['image'].copy()

  # Tokenize text and get CLIP features.
  text_tokens = clip.tokenize(text).cuda()
  with torch.no_grad():
    text_feats = clip_model.encode_text(text_tokens).float()
  text_feats /= text_feats.norm(dim=-1, keepdim=True)
  text_feats = np.float32(text_feats.cpu())

  # Normalize image and add batch dimension.
  img = obs['image'][None, ...] / 255
  img = np.concatenate((img, coords[None, ...]), axis=3)

  # Run Transporter Nets to get pick and place heatmaps.
  batch = {'img': jnp.float32(img), 'text': jnp.float32(text_feats)}
  pick_map, place_map = eval_step(optim.target, batch)
  pick_map, place_map = np.float32(pick_map), np.float32(place_map)

  # Get pick position.
  pick_max = np.argmax(np.float32(pick_map)).squeeze()
  pick_yx = (pick_max // 224, pick_max % 224)
  pick_yx = np.clip(pick_yx, 20, 204)
  pick_xyz = obs['xyzmap'][pick_yx[0], pick_yx[1]]

  # Get place position.
  place_max = np.argmax(np.float32(place_map)).squeeze()
  place_yx = (place_max // 224, place_max % 224)
  place_yx = np.clip(place_yx, 20, 204)
  place_xyz = obs['xyzmap'][place_yx[0], place_yx[1]]

  # Step environment.
  act = {'pick': pick_xyz, 'place': place_xyz}
  obs, _, _, _ = env.step(act)

  # Show pick and place action.
  plt.title(text)
  plt.imshow(prev_obs)
  plt.arrow(pick_yx[1], pick_yx[0], place_yx[1]-pick_yx[1], place_yx[0]-pick_yx[0], color='w', head_starts_at_zero=False, head_width=7, length_includes_head=True)
  plt.show()

  # Show debug plots.
  plt.subplot(1, 2, 1)
  plt.title('Pick Heatmap')
  plt.imshow(pick_map.reshape(224, 224))
  plt.subplot(1, 2, 2)
  plt.title('Place Heatmap')
  plt.imshow(place_map.reshape(224, 224))
  plt.show()

  # Show video of environment rollout.
  debug_clip = ImageSequenceClip(env.cache_video, fps=25)
  display(debug_clip.ipython_display(autoplay=1, loop=1, center=False))
  env.cache_video = []

  # Show camera image after pick and place.
  plt.subplot(1, 2, 1)
  plt.title('Before')
  plt.imshow(before)
  plt.subplot(1, 2, 2)
  plt.title('After')
  after = env.get_camera_image()
  plt.imshow(after)
  plt.show()

  # return pick_xyz, place_xyz, pick_map, place_map, pick_yx, place_yx
  return obs


# pick_xyz, place_xyz, pick_map, place_map, pick_yx, place_yx =
obs = run_cliport(obs, user_input)





## Setup SayCan

#@title LLM Cache
overwrite_cache = True
if overwrite_cache:
  LLM_CACHE = {}

# with open('filename.pickle', 'wb') as handle:
#     pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # with open('llm_cache.pickle', 'rb') as handle:
# #     b = pickle.load(LLM_CACHE)

# @title LLM Scoring

def gpt3_call(engine="text-ada-001", prompt="", max_tokens=128, temperature=0,
              logprobs=1, echo=False):
    full_query = ""
    for p in prompt:
        full_query += p
    id = tuple((engine, full_query, max_tokens, temperature, logprobs, echo))
    if id in LLM_CACHE.keys():
        print('cache hit, returning')
        response = LLM_CACHE[id]
    else:
        response = openai.Completion.create(engine=engine,
                                            prompt=prompt,
                                            max_tokens=max_tokens,
                                            temperature=temperature,
                                            logprobs=logprobs,
                                            echo=echo)
        LLM_CACHE[id] = response
    return response


def gpt3_scoring(query, options, engine="text-ada-001", limit_num_options=None, option_start="\n", verbose=False,
                 print_tokens=False):
    if limit_num_options:
        options = options[:limit_num_options]
    verbose and print("Scoring", len(options), "options")
    gpt3_prompt_options = [query + option for option in options]
    response = gpt3_call(
        engine=engine,
        prompt=gpt3_prompt_options,
        max_tokens=0,
        logprobs=1,
        temperature=0,
        echo=True, )

    scores = {}
    for option, choice in zip(options, response["choices"]):
        tokens = choice["logprobs"]["tokens"]
        token_logprobs = choice["logprobs"]["token_logprobs"]

        total_logprob = 0
        for token, token_logprob in zip(reversed(tokens), reversed(token_logprobs)):
            print_tokens and print(token, token_logprob)
            if option_start is None and not token in option:
                break
            if token == option_start:
                break
            total_logprob += token_logprob
        scores[option] = total_logprob

    for i, option in enumerate(sorted(scores.items(), key=lambda x: -x[1])):
        verbose and print(option[1], "\t", option[0])
        if i >= 10:
            break

    return scores, response


def make_options(pick_targets=None, place_targets=None, options_in_api_form=True, termination_string="done()"):
    if not pick_targets:
        pick_targets = PICK_TARGETS
    if not place_targets:
        place_targets = PLACE_TARGETS
    options = []
    for pick in pick_targets:
        for place in place_targets:
            if options_in_api_form:
                option = "robot.pick_and_place({}, {})".format(pick, place)
            else:
                option = "Pick the {} and place it on the {}.".format(pick, place)
            options.append(option)

    options.append(termination_string)
    print("Considering", len(options), "options")
    return options


query = "To pick the blue block and put it on the red block, I should:\n"
options = make_options(PICK_TARGETS, PLACE_TARGETS)
scores, response = gpt3_scoring(query, options, engine=ENGINE, limit_num_options=5, option_start='\n', verbose=True)


# @title Helper Functions

def build_scene_description(found_objects, block_name="box", bowl_name="circle"):
    scene_description = f"objects = {found_objects}"
    scene_description = scene_description.replace(block_name, "block")
    scene_description = scene_description.replace(bowl_name, "bowl")
    scene_description = scene_description.replace("'", "")
    return scene_description


def step_to_nlp(step):
    step = step.replace("robot.pick_and_place(", "")
    step = step.replace(")", "")
    pick, place = step.split(", ")
    return "Pick the " + pick + " and place it on the " + place + "."


def normalize_scores(scores):
    max_score = max(scores.values())
    normed_scores = {key: np.clip(scores[key] / max_score, 0, 1) for key in scores}
    return normed_scores


def plot_saycan(llm_scores, vfs, combined_scores, task, correct=True, show_top=None):
    if show_top:
        top_options = nlargest(show_top, combined_scores, key=combined_scores.get)
        # add a few top llm options in if not already shown
        top_llm_options = nlargest(show_top // 2, llm_scores, key=llm_scores.get)
        for llm_option in top_llm_options:
            if not llm_option in top_options:
                top_options.append(llm_option)
        llm_scores = {option: llm_scores[option] for option in top_options}
        vfs = {option: vfs[option] for option in top_options}
        combined_scores = {option: combined_scores[option] for option in top_options}

    sorted_keys = dict(sorted(combined_scores.items()))
    keys = [key for key in sorted_keys]
    positions = np.arange(len(combined_scores.items()))
    width = 0.3

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 1, 1)

    plot_llm_scores = normalize_scores({key: np.exp(llm_scores[key]) for key in sorted_keys})
    plot_llm_scores = np.asarray([plot_llm_scores[key] for key in sorted_keys])
    plot_affordance_scores = np.asarray([vfs[key] for key in sorted_keys])
    plot_combined_scores = np.asarray([combined_scores[key] for key in sorted_keys])

    ax1.bar(positions, plot_combined_scores, 3 * width, alpha=0.6, color="#93CE8E", label="combined")

    score_colors = ["#ea9999ff" for score in plot_affordance_scores]
    ax1.bar(positions + width / 2, 0 * plot_combined_scores, width, color="#ea9999ff", label="vfs")
    ax1.bar(positions + width / 2, 0 * plot_combined_scores, width, color="#a4c2f4ff", label="language")
    ax1.bar(positions - width / 2, np.abs(plot_affordance_scores), width, color=score_colors)

    plt.xticks(rotation="vertical")
    ax1.set_ylim(0.0, 1.0)

    ax1.grid(True, which="both")
    ax1.axis("on")

    ax1_llm = ax1.twinx()
    ax1_llm.bar(positions + width / 2, plot_llm_scores, width, color="#a4c2f4ff", label="language")
    ax1_llm.set_ylim(0.01, 1.0)
    plt.yscale("log")

    font = {"fontname": "Arial", "size": "16", "color": "k" if correct else "r"}
    plt.title(task, **font)
    key_strings = [key.replace("robot.pick_and_place", "").replace(", ", " to ").replace("(", "").replace(")", "") for
                   key in keys]
    plt.xticks(positions, key_strings, **font)
    ax1.legend()
    plt.show()



#@title Affordance Scoring
#@markdown Given this environment does not have RL-trained policies or an asscociated value function, we use affordances through an object detector.

def affordance_scoring(options, found_objects, verbose=False, block_name="box", bowl_name="circle", termination_string="done()"):
  affordance_scores = {}
  found_objects = [
                   found_object.replace(block_name, "block").replace(bowl_name, "bowl")
                   for found_object in found_objects + list(PLACE_TARGETS.keys())[-5:]]
  verbose and print("found_objects", found_objects)
  for option in options:
    if option == termination_string:
      affordance_scores[option] = 0.2
      continue
    pick, place = option.replace("robot.pick_and_place(", "").replace(")", "").split(", ")
    affordance = 0
    found_objects_copy = found_objects.copy()
    if pick in found_objects_copy:
      found_objects_copy.remove(pick)
      if place in found_objects_copy:
        affordance = 1
    affordance_scores[option] = affordance
    verbose and print(affordance, '\t', option)
  return affordance_scores




#@title Test
termination_string = "done()"
query = "To pick the blue block and put it on the red block, I should:\n"

options = make_options(PICK_TARGETS, PLACE_TARGETS, termination_string=termination_string)
llm_scores, _ = gpt3_scoring(query, options, verbose=True, engine=ENGINE)

affordance_scores = affordance_scoring(options, found_objects, block_name="box", bowl_name="circle", verbose=False, termination_string=termination_string)

combined_scores = {option: np.exp(llm_scores[option]) * affordance_scores[option] for option in options}
combined_scores = normalize_scores(combined_scores)
selected_task = max(combined_scores, key=combined_scores.get)
print("Selecting: ", selected_task)




"""
SayCan
Here we implement SayCan with LLM scoring and robotic affordances from ViLD (in the absence of a trained value function from an RL policy).



"""

#@title Prompt

termination_string = "done()"

gpt3_context = """
objects = [red block, yellow block, blue block, green bowl]
# move all the blocks to the top left corner.
robot.pick_and_place(blue block, top left corner)
robot.pick_and_place(red block, top left corner)
robot.pick_and_place(yellow block, top left corner)
done()

objects = [red block, yellow block, blue block, green bowl]
# put the yellow one the green thing.
robot.pick_and_place(yellow block, green bowl)
done()

objects = [yellow block, blue block, red block]
# move the light colored block to the middle.
robot.pick_and_place(yellow block, middle)
done()

objects = [blue block, green bowl, red block, yellow bowl, green block]
# stack the blocks.
robot.pick_and_place(green block, blue block)
robot.pick_and_place(red block, green block)
done()

objects = [red block, blue block, green bowl, blue bowl, yellow block, green block]
# group the blue objects together.
robot.pick_and_place(blue block, blue bowl)
done()

objects = [green bowl, red block, green block, red bowl, yellow bowl, yellow block]
# sort all the blocks into their matching color bowls.
robot.pick_and_place(green block, green bowl)
robot.pick_and_place(red block, red bowl)
robot.pick_and_place(yellow block, yellow bowl)
done()
"""

use_environment_description = False
gpt3_context_lines = gpt3_context.split("\n")
gpt3_context_lines_keep = []
for gpt3_context_line in gpt3_context_lines:
  if "objects =" in gpt3_context_line and not use_environment_description:
    continue
  gpt3_context_lines_keep.append(gpt3_context_line)

gpt3_context = "\n".join(gpt3_context_lines_keep)
print(gpt3_context)

# @title Task and Config
only_plan = False

raw_input = "put all the blocks in different corners."
config = {"pick": ["red block", "yellow block", "green block", "blue block"],
          "place": ["red bowl"]}

# raw_input = "move the block to the bowl."
# config = {'pick':  ['red block'],
#           'place': ['green bowl']}

# raw_input = "put any blocks on their matched colored bowls."
# config = {'pick':  ['yellow block', 'green block', 'blue block'],
#           'place': ['yellow bowl', 'green bowl', 'blue bowl']}

# raw_input = "put all the blocks in the green bowl."
# config = {'pick':  ['yellow block', 'green block', 'red block'],
#           'place': ['yellow bowl', 'green bowl']}

# raw_input = "stack all the blocks."
# config = {'pick':  ['yellow block', 'blue block', 'red block'],
#           'place': ['blue bowl', 'red bowl']}

# raw_input = "make the highest block stack."
# config = {'pick':  ['yellow block', 'blue block', 'red block'],
#           'place': ['blue bowl', 'red bowl']}

# raw_input = "stack all the blocks."
# config = {'pick':  ['green block', 'blue block', 'red block'],
#           'place': ['yellow bowl', 'green bowl']}

# raw_input = "put the block in all the corners."
# config = {'pick':  ['red block'],
#           'place': ['red bowl', 'green bowl']}

# raw_input = "clockwise, move the block through all the corners."
# config = {'pick':  ['red block'],
#           'place': ['red bowl', 'green bowl', 'yellow bowl']}


#@title Setup Scene
image_path = "./2db.png"
np.random.seed(2)
if config is None:
  pick_items = list(PICK_TARGETS.keys())
  pick_items = np.random.choice(pick_items, size=np.random.randint(1, 5), replace=False)

  place_items = list(PLACE_TARGETS.keys())[:-9]
  place_items = np.random.choice(place_items, size=np.random.randint(1, 6 - len(pick_items)), replace=False)
  config = {"pick":  pick_items,
            "place": place_items}
  print(pick_items, place_items)

obs = env.reset(config)

img_top = env.get_camera_image_top()
img_top_rgb = cv2.cvtColor(img_top, cv2.COLOR_BGR2RGB)
plt.imshow(img_top)

imageio.imsave(image_path, img_top)


#@title Runner
plot_on = True
max_tasks = 5

options = make_options(PICK_TARGETS, PLACE_TARGETS, termination_string=termination_string)
found_objects = vild(image_path, category_name_string, vild_params, plot_on=False)
scene_description = build_scene_description(found_objects)
env_description = scene_description

print(scene_description)

gpt3_prompt = gpt3_context
if use_environment_description:
  gpt3_prompt += "\n" + env_description
gpt3_prompt += "\n# " + raw_input + "\n"

all_llm_scores = []
all_affordance_scores = []
all_combined_scores = []
affordance_scores = affordance_scoring(options, found_objects, block_name="box", bowl_name="circle", verbose=False)
num_tasks = 0
selected_task = ""
steps_text = []
while not selected_task == termination_string:
  num_tasks += 1
  if num_tasks > max_tasks:
    break

  llm_scores, _ = gpt3_scoring(gpt3_prompt, options, verbose=True, engine=ENGINE, print_tokens=False)
  combined_scores = {option: np.exp(llm_scores[option]) * affordance_scores[option] for option in options}
  combined_scores = normalize_scores(combined_scores)
  selected_task = max(combined_scores, key=combined_scores.get)
  steps_text.append(selected_task)
  print(num_tasks, "Selecting: ", selected_task)
  gpt3_prompt += selected_task + "\n"

  all_llm_scores.append(llm_scores)
  all_affordance_scores.append(affordance_scores)
  all_combined_scores.append(combined_scores)

if plot_on:
  for llm_scores, affordance_scores, combined_scores, step in zip(
      all_llm_scores, all_affordance_scores, all_combined_scores, steps_text):
    plot_saycan(llm_scores, affordance_scores, combined_scores, step, show_top=10)

print('**** Solution ****')
print(env_description)
print('# ' + raw_input)
for i, step in enumerate(steps_text):
  if step == '' or step == termination_string:
    break
  print('Step ' + str(i) + ': ' + step)
  nlp_step = step_to_nlp(step)

if not only_plan:
  print('Initial state:')
  plt.imshow(env.get_camera_image())

  for i, step in enumerate(steps_text):
    if step == '' or step == termination_string:
      break
    nlp_step = step_to_nlp(step)
    print('GPT-3 says next step:', nlp_step)

    obs = run_cliport(obs, nlp_step)

  # Show camera image after task.
  print('Final state:')
  plt.imshow(env.get_camera_image())







## Socratic Model: VILD, GPT3, CLIPort

This implements a version of LLM planning shown in [Socratic Models](https://socraticmodels.github.io/), without the grounding, but with a scene description. For this relatively simple environment with clear robotic affordances, the scene description is generally sufficient. This mirrors the implementation attached to the paper [here](https://github.com/google-research/google-research/tree/master/socraticmodels).

#@title Prompt

gpt3_context = """
objects = [red block, yellow block, blue block, green bowl]
# move all the blocks to the top left corner.
robot.pick_and_place(blue block, top left corner)
robot.pick_and_place(red block, top left corner)
robot.pick_and_place(yellow block, top left corner)
done()

objects = [red block, yellow block, blue block, green bowl]
# put the yellow one the green thing.
robot.pick_and_place(yellow block, green bowl)
done()

objects = [yellow block, blue block, red block]
# move the light colored block to the middle.
robot.pick_and_place(yellow block, middle)
done()

objects = [blue block, green bowl, red block, yellow bowl, green block]
# stack the blocks.
robot.pick_and_place(green block, blue block)
robot.pick_and_place(red block, green block)
done()

objects = [red block, blue block, green bowl, blue bowl, yellow block, green block]
# group the blue objects together.
robot.pick_and_place(blue block, blue bowl)
done()

objects = [green bowl, red block, green block, red bowl, yellow bowl, yellow block]
# sort all the blocks into their matching color bowls.
robot.pick_and_place(green block, green bowl)
robot.pick_and_place(red block, red bowl)
robot.pick_and_place(yellow block, yellow bowl)
done()
"""

#@title Queries and Configs

only_plan = False

raw_input = "put all the blocks in different corners."
config = {'pick':  ['red block', 'yellow block', 'green block', 'blue block'],
          'place': ['red bowl']}


#@title Runner

env_description = ''
image_path = './2db.png'

np.random.seed(2)

if config is None:
  pick_items = list(PICK_TARGETS.keys())
  pick_items = np.random.choice(pick_items, size=np.random.randint(1, 5), replace=False)

  place_items = list(PLACE_TARGETS.keys())[:-9]
  place_items = np.random.choice(place_items, size=np.random.randint(1, 6 - len(pick_items)), replace=False)
  config = {'pick':  pick_items,
            'place': place_items}
  print(pick_items, place_items)
obs = env.reset(config)

img_top = env.get_camera_image_top()
img_top_rgb = cv2.cvtColor(img_top, cv2.COLOR_BGR2RGB)
plt.imshow(img_top_rgb)

imageio.imsave(image_path, img_top)

found_objects = vild(image_path, category_name_string, vild_params, plot_on=False)
scene_description = build_scene_description(found_objects)
print(scene_description)

env_description = scene_description

gpt3_prompt = gpt3_context
gpt3_prompt += "\n" + env_description + "\n"
gpt3_prompt += "# " + raw_input
response = gpt3_call(engine=ENGINE, prompt=gpt3_prompt, max_tokens=128, temperature=0)
steps_text = [text.strip().strip() for text in response["choices"][0]["text"].strip().split("#")[0].split("\n")][:-1]
print('**** Solution ****')
print(env_description)
print('# ' + raw_input)
for i, step in enumerate(steps_text):
  if step == '' or step == termination_string:
    break
  print('Step ' + str(i) + ': ' + step)
  nlp_step = step_to_nlp(step)

if not only_plan:
  print('Initial state:')
  plt.imshow(env.get_camera_image())

  for i, step in enumerate(steps_text):
    if step == '' or step == termination_string:
      break
    nlp_step = step_to_nlp(step)
    print('GPT-3 says next step:', nlp_step)

    obs = run_cliport(obs, nlp_step)

  # Show camera image after task.
  print('Final state:')
  plt.imshow(env.get_camera_image())



































