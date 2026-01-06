import json
import argparse
import os
import re
from collections import Counter
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# Load English model (for lemmatization)
nlp = spacy.load("en_core_web_sm")

# STOP_WORDS = {'rather', 'whither', 'do', 'itself', 'both', 'we', 'for', 'had', 'most', 'why', 'have', 'namely', 'did', 'it', 'onto', 'whose', 'becoming', 'yours', 'alone', 'often', 'some', 'give', 'get', 'cannot', 'per', 'now', 'beforehand', '’ve', 'few', '‘re', 'across', 'by', 'say', 'since', 'due', 'being', 'ours', 'under', 'ever', "'ve", 'or', 'around', 'anything', '’re', 'at', 'during', 'meanwhile', 'afterwards', 'must', 'against', 'am', 'our', 'never', 'only', 'more', 'formerly', 'her', 'whereafter', 'has', 'front', '’ll', 'anyway', 'sixty', 'us', 'everything', 'and', 'was', 'hundred', 'anyone', 'that', 'thru', 'here', 'others', 'if', 'nothing', 'beyond', 'put', 'whatever', 'also', 'ca', 'seemed', 'below', 'their', 'were', 'full', 'already', 'several', 'myself', 'became', 'thereupon', 'nine', 'up', 'else', 'be', 'twelve', 'no', "n't", 'somewhere', 'into', 'each', 'his', 'very', 'whereupon', 'these', 'toward', 'whereby', 'one', 'bottom', 'via', 'last', 'wherever', 'in', 'upon', 'neither', 'whether', 'how', 'than', 'next', 'nowhere', 'other', 'thence', 'is', 'nobody', 'unless', 'still', 'moreover', 'six', 'move', 'of', 'much', 'eight', 'almost', 'are', 'then', 'part', 'latterly', 'three', 'again', 'been', 'without', 'an', 'always', 'while', 'besides', 'becomes', 'otherwise', 'which', 'not', '‘s', 'quite', "'s", 'herself', 'less', 'therein', "'d", 'used', 'to', 'until', 'sometimes', 'above', 'anyhow', 'well', 'them', 'you', 'sometime', 'before', 'various', 'whenever', 'there', 'behind', 'should', 'n’t', 'regarding', 'hereafter', 'over', 'about', 'themselves', 'somehow', 'those', 'show', 'all', 'between', '‘ll', 'whence', 'five', 'where', 'please', 'every', 'along', 'make', 'name', 'such', 'first', 'take', '‘d', 'four', 'the', 'would', 're', '‘ve', 'whole', 'empty', 'down', 'himself', 'amount', 'can', 'fifteen', 'eleven', 'him', 'even', 'same', 'using', 'after', 'something', 'someone', 'n‘t', 'former', 'except', 'its', "'re", 'he', 'side', 'towards', 'fifty', 'this', 'a', 'everyone', 'seeming', 'from', 'nor', 'hereby', 'your', 'latter', 'top', 'hereupon', 'done', '‘m', 'noone', 'twenty', 'who', 'become', 'everywhere', 'wherein', 'could', 'serious', 'mostly', 'off', 'with', 'among', 'enough', 'any', 'when', 'own', 'further', 'within', 'least', 'ten', 'hers', '’d', 'throughout', 'call', 'forty', 'they', 'yet', 'doing', 'once', 'whoever', 'my', 'none', 'through', '’s', 'does', 'two', 'elsewhere', 'yourself', 'made', 'will', "'m", 'yourselves', 'may', 'back', 'another', 'amongst', 'see', 'beside', 'because', 'herein', 'she', 'as', 'indeed', 'mine', 'too', 'ourselves', "'ll", '’m', 'go', 'whom', 'really', 'together', 'out', 'many', 'either', 'anywhere', 'though', 'what'}
# STOP_WORDS = {}
def clean_and_tokenize(text):
    """
    Tokenize + lemmatize + remove stop words + filter non-letters.
    """
    # Optionally clean newlines and special characters
    # text = re.sub(r'\s+', ' ', text)  # Replace all whitespace with a single space
    
    doc = nlp(text.lower())

    tokens = [
        token.text for token in doc
        # if token.is_alpha and token.lemma_ not in STOP_WORDS
        # and token.lemma_ not in STOP_WORDS
        # if token.text.strip()  # Ensure token is not whitespace
        # token.lemma_
        # for token in doc
    ]
    return tokens


def extract_code_texts(path):
    """Extract code field text from jsonl (skip bad lines and log to bad_lines.log)."""
    all_texts = []
    log_path = os.path.join(os.path.dirname(path), "bad_lines.log")

    # Append to log to collect errors from multiple files
    with open(log_path, "a", encoding="utf-8") as log_f:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):

                clean = line.strip()
                if not clean:
                    continue

                try:
                    obj = json.loads(clean)
                except json.JSONDecodeError as e:
                    # Write to bad_lines.log
                    log_f.write(
                        f"[BAD JSON] file={path}, line={line_num}\n"
                        f"  error: {str(e)}\n"
                        f"  content: {repr(clean[:500])}\n"
                        f"--------------------------------------------------\n"
                    )
                    print(f"[WARN] Skipped bad line: {path} (line {line_num}); see bad_lines.log")
                    continue

                codes = obj.get("code", [])
                all_texts.extend(codes)

    return all_texts


def count_ngrams(tokens, n=1):
    """Count n-grams."""
    if n == 1:
        return Counter(tokens)
    return Counter([" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])
    # Clean special characters in n-grams
    # ngrams = []
    # for i in range(len(tokens) - n + 1):
    #     ngram = " ".join(tokens[i:i+n])
    #     # Remove possible newlines and extra spaces
    #     ngram = re.sub(r'\s+', ' ', ngram).strip()
    #     if ngram:  # Only add non-empty n-grams
    #         ngrams.append(ngram)
    
    # return Counter(ngrams)


def filter_low_freq(counter, min_count=2):
    """Filter out low-frequency terms/phrases."""
    return Counter({k: v for k, v in counter.items() if v >= min_count})


def compare_freq(f1, f2):
    """Directional comparison."""
    all_keys = set(f1) | set(f2)

    file1_higher = []
    file2_higher = []

    for term in all_keys:
        c1, c2 = f1.get(term, 0), f2.get(term, 0)

        if c1 > c2:
            file1_higher.append(
                {"term": term, "file1": c1, "file2": c2, "diff": c1 - c2, "ratio": (c1+1)/(c2+1)}
            )
        elif c2 > c1:
            file2_higher.append(
                {"term": term, "file1": c1, "file2": c2, "diff": c2 - c1, "ratio": (c2+1)/(c1+1)}
            )

    file1_higher.sort(key=lambda x: x["diff"], reverse=True)
    file2_higher.sort(key=lambda x: x["diff"], reverse=True)

    return file1_higher, file2_higher


def plot_bar(data, title, out_path, top_k=20):
    """Plot bar chart."""
    def sanitize_label(label):
        text = str(label).replace("\n", " ").replace("\r", " ")
        return text.replace("$", r"\$")

    words = [sanitize_label(d["term"]) for d in data[:top_k]]
    values = [d["diff"] for d in data[:top_k]]

    plt.figure(figsize=(12, 6))
    plt.barh(words[::-1], values[::-1])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved bar chart:", out_path)


# def plot_wordcloud(counter, out_path):
#     """Plot word cloud."""
#     wc = WordCloud(width=1000, height=600, background_color="white")
#     wc.generate_from_frequencies(counter)
#     wc.to_file(out_path)
#     print("Saved word cloud:", out_path)


def plot_wordcloud(counter, out_path):
    """Plot word cloud."""
    if not counter:
        print(f"Warning: Empty counter for {out_path}, skipping...")
        return
    
    # Clean word cloud data and remove words with newlines
    cleaned_counter = {}
    for word, freq in counter.items():
        # Remove newlines and extra spaces
        cleaned_word = re.sub(r'\s+', ' ', str(word)).strip()
        if cleaned_word and '\n' not in cleaned_word:
            cleaned_counter[cleaned_word] = freq
    
    if not cleaned_counter:
        print(f"Warning: No valid words for {out_path} after cleaning, skipping...")
        return
    
    wc = WordCloud(
        width=1000, 
        height=600, 
        background_color="white",
        max_words=200,
        relative_scaling=0.5,
        min_font_size=10
    )
    wc.generate_from_frequencies(cleaned_counter)
    wc.to_file(out_path)
    print("Saved word cloud:", out_path)

def generate_style_summary(title, terms):
    """Auto-generate style summary text."""
    top_terms = [t["term"] for t in terms[:15]]
    summary = f"Common reasoning catchphrases used more often by the {title} model include:\n"
    summary += ", ".join(top_terms)
    summary += ".\nThese words/phrases reflect the model's typical reasoning style.\n"
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file1", required=True)
    parser.add_argument("--file2", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--min_count", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ========== Process both files ==========
    def process(path):
        texts = extract_code_texts(path)
        tokens = []
        for t in texts:
            tokens.extend(clean_and_tokenize(t))

        ngrams = {
            "unigram": filter_low_freq(count_ngrams(tokens, 1), args.min_count),
            "bigram": filter_low_freq(count_ngrams(tokens, 2), args.min_count),
            "trigram": filter_low_freq(count_ngrams(tokens, 3), args.min_count),
            "fourgram": filter_low_freq(count_ngrams(tokens, 4), args.min_count),
            "fivegram": filter_low_freq(count_ngrams(tokens, 5), args.min_count),
        }
        return ngrams

    stats1 = process(args.file1)
    stats2 = process(args.file2)

    results = {}
    style_summary = ""

    for gram in ["unigram", "bigram", "trigram", "fourgram", "fivegram"]:
        f1_higher, f2_higher = compare_freq(stats1[gram], stats2[gram])

        # Save structured data
        results[gram] = {
            "file1_higher": f1_higher[:args.top_k],
            "file2_higher": f2_higher[:args.top_k],
        }

        # Visualization
        plot_bar(f1_higher, f"{gram}: File1 > File2", os.path.join(args.output_dir, f"{gram}_file1_bar.png"))
        plot_bar(f2_higher, f"{gram}: File2 > File1", os.path.join(args.output_dir, f"{gram}_file2_bar.png"))

        # Word cloud
        plot_wordcloud(dict(stats1[gram]), os.path.join(args.output_dir, f"{gram}_file1_wc.png"))
        plot_wordcloud(dict(stats2[gram]), os.path.join(args.output_dir, f"{gram}_file2_wc.png"))

        # Auto style summary
        style_summary += generate_style_summary("File 1", f1_higher)
        style_summary += "\n"
        style_summary += generate_style_summary("File 2", f2_higher)
        style_summary += "\n\n"

    # Write JSON
    with open(os.path.join(args.output_dir, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Write text summary
    with open(os.path.join(args.output_dir, "style_summary.txt"), "w", encoding="utf-8") as f:
        f.write(style_summary)

    print("\nAnalysis completed!")
    print("Results saved to:", args.output_dir)


if __name__ == "__main__":
    main()
