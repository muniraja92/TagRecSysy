Here is the code with all non-English text removed and cleaned up:

```python
import os
import pandas as pd
import tqdm
import np
import argparse
import random
import sys
import openai

from langchain import OpenAI, ConversationChain
from langchain.prompts import PromptTemplate
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer


def format_data(data, preference):
    examples = [
        "\"For example, given a video, its \"title\" is \"The cage is quite high, eighty percent won't jump out, don't know if it was eaten by something, but no blood was seen, continue searching for biubiu, do you have any small methods to find hamsters?\", \"category\" is \"animals\", \"ocr\" is \"Today I found that biubiu is gone, biubiu is nowhere, the cage cover was not closed last night\", \"asr\" is \"Today I found that BB I was gone, BB is nowhere, the cage cover was not closed last night, it should have run out, but this cage is high, usually can't get out, keep looking in area B.\". The robot infers the reasonable tags \"hamster pseudo hibernation, hamster feigned death, hamster missing, hamster hibernation\".",
        "\"For example, given a video, its \"title\" is \"Can't draw anime legs? Check if your drawing is correct #anime #hand drawing tutorial #hand drawing #future designer\", \"category\" is \"talent\", \"ocr\" is \"Can't draw nice anime legs, learn to draw legs with me like this\", \"asr\" is \"So stay in the other small road see if I look like it, I have to do so every day like a walk will forget being in a fairy jumping out of a mysterious mood, want to say to the whole world, so stay in the other small road see if I look like it? I and have to do so every day like a walk will forget.\". The robot infers the reasonable tags \"anime teacher, anime character drawing tutorial, how to draw comic legs, beginner's drawing tutorial\".",
        "\"For example, given a video, its \"title\" is \"Daily life hacks #life hacks #Inner Mongolia specialty\", \"category\" is \"health, life\", \"ocr\" is \"life hacks\", \"asr\" is \"Knowing these small tricks, you've found a treasure, when inserting a straw it bends easily, just seal the top with your thumb, then you can easily cut the straw with scissors, like this cut it will easily take out the hair in the drain, use a small knife to cut the watermelon in a spiral also look at it to store home wires batteries too convenient four straws cut off both ends. Leave the small spring in the middle, seal the unfinished yogurt very practical, follow me, learn more about small life experiments.\". The robot infers the reasonable tags \"daily life hacks, life hacks, comprehensive list of small hacks\".",
        "\"For example, given a video, its \"title\" is \"Changan's cheapest sedan, starts at 40k many people look down on it, but I know a car is just a means of transportation, why need any face! #ChanganCars\", \"category\" is \"cars\", \"ocr\" is \"The cheapest Changan sedan\", \"asr\" is \"I don't deny that there is still a gap between domestic and joint venture cars, but indeed it is they who let us drive 50k opened MP V8 100k opened up the sedan, 150k opened up a large seven-seater.\". The robot infers the reasonable tags \"Changan sedan price, cheapest Changan sedan, new Changan sedan\".",
        "\"For example, given a video, its \"title\" is \"Whole house embedded subwoofer, mainly this projector really is love 💕\", \"category\" is \"real estate/home\", \"ocr\" is \"42 sqm, one-bedroom small apartment\", \"asr\" is \"Look, lights in the distance shining bright. You alone bow your head on the road. The bigger the city, the more it makes one anxious more longing, more prolonged. Wish a journey too many injuries. Forget the initial smile. Time makes us fragile, yet strong, let me in love Qingqing sing to you. I wish I could accompany you more to sing. Tell you the landscape of life.\". The robot infers the reasonable tags \"small apartment decoration, one-bedroom decoration, decoration effect pictures\"."
    ]
    sentences = []
    prompt = PromptTemplate(
        input_variables=["preference", "caption", "ocr_cover", "asr_pure", "category_name", "example"],
        template="You are a video {preference} generation robot, according to the input video title, category, ocr, asr to reason out the reasonable \"{

preference}\", expressed in the form of multiple labels separated by commas. {example} So, given a new video, its \"title\" is \"{caption}\", \"category\" is \"{category_name}\", \"ocr\" is \"{ocr_cover}\", \"asr\" is \"{asr_pure}\", please infer the video's \"{preference}\":"
    )
    for ind, row in enumerate(tqdm.tqdm(data.iterrows())):
        example = examples[random.randint(0, 4)]
        caption = row[1]['caption'][:100]
        ocr_cover = row[1]['ocr_cover'][:100]
        asr_pure = row[1]['asr_pure'][:100]
        text = prompt.format(
            preference=preference,
            caption=caption,
            category_name=row[1]['category_name'],
            ocr_cover=ocr_cover,
            asr_pure=asr_pure, example=example
        )

        sentences.append(text)

    f = open('../data/sentences.txt', 'w')
    f.write("\n".join(sentences))
    f.close()


def tag_gen(data_path, openai_key, gen_feq):
    openai.api_key = openai_key

    sentences = []
    f = open(data_path, 'r')
    for line in f.readlines():
        sentences.append(line.strip())
    f.close()

    num = 0
    final_res = []
    for sentence in tqdm.tqdm(sentences):
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": sentence}],
                temperature=1.5,
                n=gen_feq
            )

            res = str(num) + "||"
            for j in range(gen_feq):
                ans = completion.choices[j].message["content"].strip()
                ans = ans.replace("\n", "")
                res += str(ans) + "||"

            final_res.append(res)
        except:
            continue

        num += 1
        if len(final_res) == 100:
            f = open("../data/tag_gen.txt", 'a')
            f.write("\n".join(final_res))
            f.close()
            final_res = []


def posterior_process(data_path):
    f = open(data_path, 'r')
    out = ""
    tag_all = []
    for line in f.readlines():
        line = line.replace(".", "")
        line = line.replace("。", "")
        line = line.replace(",", "、")
        line = line.replace("，", "、")
        line = line.replace("'", "")
        line = line.replace("\n", "")
        line = line.replace("\"", "")
        tmp = line.strip().split('||')
        out += str(tmp) + "\n"
        for t in tmp:
            if '、' in t:
                tags = t.split('、')
                tag_all += tags
    f.close()

    ans = Counter(tag_all)
    ans = sorted(ans.items(), key=lambda x: x[1], reverse=True)

    tags = []
    for tmp in ans:
        if tmp[1] > 2:
            tags.append(tmp[0].replace(' ', ''))

    f = open('../data/tags.txt', 'w')
    f.write('\n'.join(tags))
    f.close()

    encoder = SentenceTransformer('hfl/chinese-roberta-wwm-ext-large')
    tags_embed = encoder.encode(tags)
    tags_dis = [np.sqrt(np.dot(_, _.T)) for _ in tags_embed]
    mark = [0 for _ in range(len(tags))]
    include = [[] for _ in range(len(tags))]

    for i in tqdm.trange(len(tags)):
        if mark[i] == 0:
            score = np.dot(tags_embed[i], tags_embed[i:].T)
            for j in range(i, len(tags)):
                if i != j:
                    score[j - i] = score[j - i] / (tags_dis[i] * tags_dis[j])
                    if score[j - i] > 0.95:
                        mark[j] = 1
                        include[i].append(tags[j])

    out = ""
    for i in range(len(tags)):
        if mark[i] == 0:
            out += tags[i] + "||" + str(include[i]) + "\n"

    f = open('../data/final_tags.csv', 'w')
    f.write(out)
    f.close()


def get_tag_embed(encoder, tags):
    tags_embed = encoder.encode(tags)
    tags_dis = [np.sqrt(np.dot(_, _.T)) for _ in tags_embed]

    with open('../data/tags_embed.npy', 'wb') as f:
        np.save(f, tags_embed)

    with open('../data/tags_dis.npy', 'wb') as f:
        np.save(f, tags_dis)

    return

 tags_embed, tags_dis


def load_tag_embed():
    tags_embed = np.load('../data/tags_embed.npy')
    tags_dis = np.load('../data/tags_dis.npy')

    return tags_embed, tags_dis


def format_prompt_selective(data, candidate_tags):
    preference = "interest tags"

    examples_tags = [
        ['hamster maze', 'hamster biting', 'catching mice', 'lice removal', 'cat and mouse', 'cat searching', 'lip hair removal', 'animal life', 'animal birth', 'eliminating mice', 'cage', 'rat control', 'zoology', 'animal mukbang', 'animal production', 'fish trap fishing', 'pet cages', 'animal teaching', 'animal PK', 'bait atomization', 'hamster illness', 'cage training', 'golden bear cage cleaning', 'cage cleaning', 'animals', 'animal show', 'hamster cage', 'transparent cage', 'hamster exercise', 'removing mustache', 'piglet diarrhea', 'soup dumplings', 'animal ecology', 'eliminating cockroaches'],
        ['forget-me-not bouquet', 'beautiful legs secrets', 'leg shape correction', 'learning simple drawings', 'drawing teaching', 'self-taught painting', 'personal talent', 'slim legs exercise', 'entertainment talent', 'quick hand talent', 'easy learning drawing method', 'devil training camp', 'talent award', 'mini world', 'talent expression', 'talent contestant', 'simple drawing techniques', 'paper art talent', 'perfect leg shape', 'slim legs plan', 'mountain bike speed descent', 'talent', 'beautiful legs exercise', 'comic teaching', 'leg health', 'anime character drawing tutorial', 'suppression fishing rod', 'high heels simple drawing tutorial', 'leg shape improvement', 'night demon', 'talent project', 'leg shape assessment', 'drawing skills', 'extreme companionship on earth', 'creative talent', 'skirt drawing skills', 'beautiful legs', 'hand magician', 'Wei Long konjac cool', 'Bai Li Xuan Ce'],
        ['bending skills', 'pruning skills', 'women's life hacks', 'life hacks', 'life skills', 'protective bandage', 'small hacks', 'life health', 'reinforcement tying skills', 'pear tree pruning skills', 'life conditioning', 'broken thread remover', 'nail trimming skills', 'health', 'car hacks', 'summer life hacks', 'health hacks', 'scissor noodles', 'steel cable socket', 'healthy life', 'life common sense', 'everyday life', 'small well-off life', 'home hacks', 'physiological health', 'creative life hacks', 'life hacks', 'daily skills', 'health and wellness', 'learning hacks', 'positive energy life hacks', 'mobile hacks', 'manual pipe bender', 'life wellness', 'life', 'DIY hacks', 'fitness life'],
        ['cheap cars', 'car accessories', 'luxury sedan recommendations', 'entry-level luxury cars', 'car science part two', 'highest cost-performance cars', 'cheap sports cars', 'luxury car market', 'car boutique', 'entry-level SUV', 'domestic luxury sedans', 'luxury seven-seater SUVs', 'car manufacturing', '100k level SUV', 'car electronics', 'car delivery', 'cheapest van', 'Wei brand cars', 'cost-effective car brands', 'SUV cars', 'cheap SUVs', 'economical and practical sedans', 'car DIY', 'seven-seater car recommendations', 'cheap good cars', 'cost-effective SUVs', 'cars', 'world's most expensive car', 'cost-effective sedan recommendations', 'cost-effective sports cars', 'luxury SUV buying guide', 'luxury ride'],
        ['car audio systems', 'real estate/home', 'house types', 'audio tuning', 'car audio', 'one-story house design', 'peaceful years', 'JBL audio', 'smart speaker', 'small kitchen', 'small furniture', 'roadside scenery', 'small bathroom design', 'small apartment decoration', 'Nordic home', 'home goods', 'youthful years', 'small space utilization', 'home electrics', 'journey of the heart', 'audio', 'home DIY', '100 sqm decoration', 'night driving light operation', 'car audio installation', 'small apartment', 'home building materials', 'practical home', 'long-distance driving', 'youth forever', 'chasing dreams', 'BOSE audio', 'passage of time', 'home life', 'audio modification', 'campus time', 'home wear', 'home', 'audio setup', 'three-room two-hall decoration']
    ]
    examples = [
        "For example, given a video, its \"title\" is \"The cage is quite high, eighty percent won't jump out, don't know if it was eaten by something,

 but no blood was seen, continue searching for biubiu, do you have any small methods to find hamsters?\", \"category\" is \"animals\", \"ocr\" is \"Today I found that biubiu is gone, nowhere to be found, the cage cover was not closed last night\", \"asr\" is \"Today I found that BB I was gone, nowhere to be found, the cage cover was not closed last night, it should have run out, but this cage is high, usually can't get out, keep looking in area B.\". The robot infers the reasonable tags from the set \"animal life, hamster cage, pet cage, hamster illness\"."
        "".format(preference, ', '.join(examples_tags[0]), preference),
        "For example, given a video, its \"title\" is \"Can't draw anime legs? Check if your drawing is correct #anime #hand drawing tutorial #hand drawing #future designer\", \"category\" is \"talent\", \"ocr\" is \"Can't draw nice anime legs, learn to draw legs with me like this\", \"asr\" is \"So stay in the other small road see if I look like it, I have to do so every day like a walk will forget being in a fairy jumping out of a mysterious mood, want to say to the whole world, so stay in the other small road see if I look like it? I and have to do so every day like a walk will forget.\". The robot infers the reasonable tags from the set \"learning simple drawings, drawing teaching, self-taught painting, simple drawing techniques, drawing skills, perfect leg shape, anime character drawing tutorial, comic teaching\"."
        "".format(preference, ', '.join(examples_tags[1]), preference),
        "For example, given a video, its \"title\" is \"Daily life hacks #life hacks #Inner Mongolia specialty\", \"category\" is \"health, life\", \"ocr\" is \"life hacks\", \"asr\" is \"Knowing these small tricks, you've found a treasure, when inserting a straw it bends easily, just seal the top with your thumb, then you can easily cut the straw with scissors, like this cut it will easily take out the hair in the drain, use a small knife to cut the watermelon in a spiral also look at it to store home wires batteries too convenient four straws cut off both ends. Leave the small spring in the middle, seal the unfinished yogurt very practical, follow me, learn more about small life experiments.\". The robot infers the reasonable tags from the set \"life hacks, life skills, small hacks, life common sense, home hacks, creative life hacks, life hacks, learning hacks, DIY hacks\"."
        "".format(preference, ', '.join(examples_tags[2]), preference),
        "For example, given a video, its \"title\" is \"Changan's cheapest sedan, starts at 40k many people look down on it, but I know a car is just a means of transportation, why need any face! #ChanganCars\", \"category\" is \"cars\", \"ocr\" is \"The cheapest Changan sedan\", \"asr\" is \"I don't deny that there is still a gap between domestic and joint venture cars, but indeed it is they who let us drive 50k opened MP V8 100k opened up the sedan, 150k opened up a large seven-seater.\". The robot infers the reasonable tags from the set \"cheap cars, highest cost-performance cars, cheapest van, cost-effective car brands, economical and practical sedans, cheap good cars, cost-effective sedan recommendations\"."
        "".format(preference, ', '.join(examples_tags[3]), preference),
        "For example, given a video, its \"title\" is \"Whole house embedded subwoofer, mainly this projector really is love 💕\", \"category\" is \"real estate/home\", \"ocr\" is \"42 sqm, one-bedroom small apartment\", \"asr\" is \"Look, lights in the distance shining bright. You alone bow your head on the road. The bigger the city, the more it makes one anxious more longing, more prolonged. Wish a journey too many injuries. Forget the initial smile. Time makes us fragile, yet strong, let me in love Qingqing sing to you. I wish I could accompany you more to sing. Tell you the landscape of life.\". The robot infers the reasonable tags from the set \"house types, small furniture, audio tuning, small apartment decoration, small space utilization, small apartment, home life\"."
        "".format(preference, ', '.join(examples_tags[4]), preference)
    ]

    prompt = PromptTemplate(
        input_variables=["preference", "caption", "ocr", "asr", "category_name", "example", "candidate_tags"],
        template="You are a video {preference} generation robot, according to the input

 video title, category, ocr, asr to reason out the reasonable \"{preference}\", expressed in the form of multiple labels separated by commas. {example} So, given a new video, its \"title\" is \"{caption}\", \"category\" is \"{category_name}\", \"ocr\" is \"{ocr}\", \"asr\" is \"{asr}\", please infer the video's \"{preference}\" from the set \"{candidate_tags}\":"
    )

    example = examples[random.randint(0, 4)]
    data_d = data.to_dict()
    text = prompt.format(preference=preference, caption=data_d['caption'],
                         category_name=data_d['category_name'], ocr=data_d['ocr'],
                         asr=data_d['asr'], example=example, candidate_tags=", ".join(candidate_tags))

    return text


def selective_tagger(data_path, tag_path, api_key):
    openai.api_key = api_key

    df_exp = pd.read_csv(data_path, sep='||', on_bad_lines='skip')
    df_tag = pd.read_csv(tag_path, sep='||', on_bad_lines='skip')
    df_tag.columns = ['tag', 'contain_tags']
    tags = list(df_tag['tag'])

    encoder = SentenceTransformer('hfl/chinese-roberta-wwm-ext-large')
    if os.path.exists('../data/tags_dis.npy') and os.path.exists('../data/tags_embed.npy'):
        tags_embed, tags_dis = load_tag_embed()
    else:
        print("Generating tag embedding")
        tags_embed, tags_dis = get_tag_embed(encoder, tags)

    selective_tags = []
    for ind, row in enumerate(tqdm.tqdm(df_exp.iterrows())):
        inputs = [row[1]['caption'], row[1]['category_name'], row[1]['ocr'], row[1]['asr']]
        input_embed = encoder.encode(inputs)
        input_dis = [np.sqrt(np.dot(_, _.T)) for _ in input_embed]

        ans = np.dot(input_embed, tags_embed.T)
        for i in range(ans.shape[0]):
            for j in range(ans.shape[1]):
                ans[i][j] = ans[i][j] / (input_dis[i] * tags_dis[j])

        candidate_tags = []
        for i in range(ans.shape[0]):
            tmp = [_ for _ in zip(list(ans[i]), tags)]
            tmp = sorted(tmp, key=lambda x: x[0], reverse=True)
            candidate_tags += [_[1] for _ in tmp[:10]]

        candidate_tags = list(set(candidate_tags))
        text = format_prompt_selective(row[1], candidate_tags)

        final_res = []
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": text}],
                temperature=1.5,
                n=5
            )

            res = []
            for j in range(5):
                ans = completion.choices[j].message["content"].strip()
                ans = ans.replace("\n", "")
                ans = ans.replace("。", "")
                ans = ans.replace("，", "、")
                res += ans.split('、')

            final_res += res
            tag_count = defaultdict(int)
            for fr in final_res:
                if fr in candidate_tags:
                    tag_count[fr] += 1

            tag_count = sorted(tag_count.items(), key=lambda x: x[1], reverse=True)
        except:
            tag_count = []
            print("api error")

        selective_tags.append(tag_count)

    return selective_tags


def generative_tagger(data_path, tag_path, api_key):
    openai.api_key = api_key

    df_exp = pd.read_csv(data_path, sep='||', on_bad_lines='skip')
    df_tag = pd.read_csv(tag_path, sep='||', on_bad_lines='skip')
    df_tag.columns = ['tag', 'contain_tags']
    tags = list(df_tag['tag'])

    encoder = SentenceTransformer('hfl/chinese-roberta-wwm-ext-large')
    if os.path.exists('../data/tags_dis.npy') and os.path.exists('../data/tags_embed.npy'):
        tags_embed, tags_dis = load_tag_embed()
    else:
        print("Generating tag embedding")
        tags_embed, tags_dis = get_tag_embed(encoder, tags)

    preference = "interest tags"
    examples = [
        "For example, given a video, its \"title\" is \"The cage is quite high, eighty percent won't jump out, don't know if it was eaten by something, but no blood was seen, continue searching for biubiu, do you have any small methods to find hamsters?\", \"category\" is \"animals\", \"ocr\" is \"Today I found that biubiu is gone, nowhere to be found, the cage cover was not closed last night

\", \"asr\" is \"Today I found that BB I was gone, nowhere to be found, the cage cover was not closed last night, it should have run out, but this cage is high, usually can't get out, keep looking in area B.\". The robot infers the reasonable tags \"hamster pseudo hibernation, hamster feigned death, hamster missing, hamster hibernation\".",
        "For example, given a video, its \"title\" is \"Can't draw anime legs? Check if your drawing is correct #anime #hand drawing tutorial #hand drawing #future designer\", \"category\" is \"talent\", \"ocr\" is \"Can't draw nice anime legs, learn to draw legs with me like this\", \"asr\" is \"So stay in the other small road see if I look like it, I have to do so every day like a walk will forget being in a fairy jumping out of a mysterious mood, want to say to the whole world, so stay in the other small road see if I look like it? I and have to do so every day like a walk will forget.\". The robot infers the reasonable tags \"anime teacher, anime character drawing tutorial, how to draw comic legs, beginner's drawing tutorial\".",
        "For example, given a video, its \"title\" is \"Daily life hacks #life hacks #Inner Mongolia specialty\", \"category\" is \"health, life\", \"ocr\" is \"life hacks\", \"asr\" is \"Knowing these small tricks, you've found a treasure, when inserting a straw it bends easily, just seal the top with your thumb, then you can easily cut the straw with scissors, like this cut it will easily take out the hair in the drain, use a small knife to cut the watermelon in a spiral also look at it to store home wires batteries too convenient four straws cut off both ends. Leave the small spring in the middle, seal the unfinished yogurt very practical, follow me, learn more about small life experiments.\". The robot infers the reasonable tags \"daily life hacks, life hacks, comprehensive list of small hacks\".",
        "For example, given a video, its \"title\" is \"Changan's cheapest sedan, starts at 40k many people look down on it, but I know a car is just a means of transportation, why need any face! #ChanganCars\", \"category\" is \"cars\", \"ocr\" is \"The cheapest Changan sedan\", \"asr\" is \"I don't deny that there is still a gap between domestic and joint venture cars, but indeed it is they who let us drive 50k opened MP V8 100k opened up the sedan, 150k opened up a large seven-seater.\". The robot infers the reasonable tags \"Changan sedan price, cheapest Changan sedan, new Changan sedan\".",
        "For example, given a video, its \"title\" is \"Whole house embedded subwoofer, mainly this projector really is love 💕\", \"category\" is \"real estate/home\", \"ocr\" is \"42 sqm, one-bedroom small apartment\", \"asr\" is \"Look, lights in the distance shining bright. You alone bow your head on the road. The bigger the city, the more it makes one anxious more longing, more prolonged. Wish a journey too many injuries. Forget the initial smile. Time makes us fragile, yet strong, let me in love Qingqing sing to you. I wish I could accompany you more to sing. Tell you the landscape of life.\". The robot infers the reasonable tags \"small apartment decoration, one-bedroom decoration, decoration effect pictures\"."
    ]

    prompt = PromptTemplate(
        input_variables=["preference", "caption", "ocr", "asr", "category_name", "example"],
        template="You are a video {preference} generation robot, according to the input video title, category, ocr, asr to reason out the reasonable \"{preference}\", expressed in the form of multiple labels separated by commas. {example} So, given a new video, its \"title\" is \"{caption}\", \"category\" is \"{category_name}\", \"ocr\" is \"{ocr}\", \"asr\" is \"{asr}\", please infer the video's \"{preference}\"."
    )

    final_res = []
    for ind, row in enumerate(tqdm.tqdm(df_exp.iterrows())):
        data = row[1].to_dict()
        example = examples[random.randint(0, 4)]
        text = prompt.format(preference=preference, caption=data['caption'],
                             category_name=data['category_name'], ocr=data['ocr'],
                             asr=data['asr'], example=example)

        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "

content": text}],
                temperature=1.5,
                n=5
            )

            res = []
            for j in range(5):
                ans = completion.choices[j].message["content"].strip()
                ans = ans.replace("\n", "")
                ans = ans.replace("。", "")
                ans = ans.replace("，", "、")
                res += ans.split('、')

            tag_count = Counter(res)
            tag_count = sorted(tag_count.items(), key=lambda x: x[1], reverse=True)

            candidate_tags = [_[0] for _ in tag_count]
            candidate_tags_embed = encoder.encode(candidate_tags)
            candidate_tags_dis = [np.sqrt(np.dot(_, _.T)) for _ in candidate_tags_embed]

            scores = np.dot(candidate_tags_embed, tags_embed.T)

            ans = []
            for i in range(scores.shape[0]):
                for j in range(scores.shape[1]):
                    score = scores[i][j] / (candidate_tags_dis[i] * tags_dis[j])
                    if score > 0.9:
                        ans.append(tags[j])

            ans = Counter(ans)
            ans = sorted(ans.items(), key=lambda x: x[1], reverse=True)

            final_res.append(ans)
        except:
            print("api error")

    return final_res


class Data:
    def __init__(self, path):
        this.path = path
        this.dataframe = this.data_loader()

    def data_loader(self):
        df = pd.read_feather(this.path)
        df_f = df[['item_id', 'caption', 'ocr_cover', 'asr_pure', 'category_name']]

        return df_f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="data path", default="")
    parser.add_argument("--tag_path", type=str, help="tag path", default="")
    parser.add_argument("--func", type=str, help="func", default="")
    parser.add_argument("--openai_key", type=str, help="openai key", default="")
    parser.add_argument("--gen_feq", type=int, help="gen_feq", default=5)

    paras = parser.parse_args()

    data_path = paras.data_path
    tag_path = paras.tag_path
    func = paras.func
    gen_feq = paras.gen_feq
    openai_key = paras.openai_key

    if func == "data_format":
        format_data(data=Data(path=data_path).dataframe, preference="interest tags")
        print("Data formatting completed")
    elif func == "tag_gen":
        tag_gen(data_path, openai_key, gen_feq)
        print("Tag generation completed")
    elif func == "posterior_process":
        posterior_process(data_path)
        print("Posterior processing completed")
    elif func == "selective_tagger":
        results = selective_tagger(data_path, tag_path, openai_key)
        print("Tagging completed")
        print(results)
    elif func == "generative_tagger":
        results = generative_tagger(data_path, tag_path, openai_key)
        print("Tagging completed")
        print(results)


if __name__ == "__main__":
    main()
```
