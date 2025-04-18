import os
import gc
import pickle
import torch
import numpy as np
import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from transformers import AutoTokenizer, AutoModel
import json
import subprocess
from Bio import SeqIO
import logging
import datetime
import shutil

USER_NAME = "hisahiro_ikari"
    
# 環境に合わせた各種パスの設定（必要に応じて変更してください）
HMMER3_BINDIR = f"/home/{USER_NAME}/hmmer/bin"        # HMMER3 のバイナリディレクトリ
DB_DIR = f"/work/{USER_NAME}/public_databases"         # 配列・構造データベースのディレクトリ
MODEL_DIR = f"/home/{USER_NAME}/models"                 # モデルパラメータのディレクトリ
ALPHAFOLD3DIR = f"/home/{USER_NAME}/alphafold3" # AlphaFold3 のコードが格納されたディレクトリ
OUTPUT_DIR = "output"
def get_checkpoint_dir(base_path, base_name):
    # 今日の日付を "YYYYMMDD" の形式で取得
    today_str = datetime.datetime.now().strftime("%Y%m%d")
    # 基本のディレクトリ名を組み立てる
    checkpoint_dir = os.path.join(base_path, f"{base_name}{today_str}")
    
    # 同名ディレクトリが存在する場合は、連番のサフィックスを追加して新たなディレクトリ名を作成
    suffix = 1
    original_dir = checkpoint_dir
    while os.path.exists(checkpoint_dir):
        checkpoint_dir = f"{original_dir}_{suffix}"
        suffix += 1
    
    return checkpoint_dir

def get_first_protein_sequence(fasta_file_path):
    """FASTAファイルから最初のタンパク質の配列を取得して返す

    Parameters:
        fasta_file_path (str): FASTAファイルのパス

    Returns:
        str: 最初のレコードのタンパク質配列（文字列形式）
        
    Raises:
        ValueError: ファイル内に配列が見つからなかった場合
    """
    with open(fasta_file_path, 'r') as handle:
        # FASTAファイル内のレコードはジェネレーターとして取得されるため、
        # next() を使って最初のレコードを取り出す
        record = next(SeqIO.parse(handle, "fasta"), None)
        if record is None:
            raise ValueError("FASTAファイルに配列が含まれていません。")
        return str(record.seq)
    
def create_alphafold_dict(given_sequence0, given_sequence1, output_name):
    return {
            "name": output_name,
            "modelSeeds": ["42"],
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": given_sequence0,
                        "count": 1
                    }
                },
                {
                    "proteinChain": {
                        "sequence": given_sequence1,
                        "count": 1
                    }
                }
            ]
        }
def create_json(given_sequence0, given_sequence1, output_name, output_dir):
    # JSON形式のデータを定義
    os.makedirs(output_dir, exist_ok = True)
    output_path = os.path.join(output_dir, f"{output_name}.json")
    data = [
        create_alphafold_dict(given_sequence0,given_sequence1,output_name)
    ]
    
    # 指定のファイルパスにJSONデータを書き込み
    with open(output_path, "w") as outfile:
        json.dump(data, outfile, indent=4)

def create_json_3J0A(given_sequence, output_name, output_dir):
    file_B = "fasta_files/3J0A_single.fasta"
    seq_raw = get_first_protein_sequence(file_B)
    create_json(given_sequence, seq_raw, output_name, output_dir)

def create_multiple_json_3J0A(given_sequences, output_name, output_dir):
    os.makedirs(output_dir, exist_ok = True)
    file_B = "fasta_files/3J0A_single.fasta"
    seq_raw = get_first_protein_sequence(file_B)
    data = [create_alphafold_dict(seq_raw, given_seq, f"{output_name}_{i}") for i, given_seq in enumerate(given_sequences)]
    output_path = os.path.join(output_dir, f"{output_name}.json")
    with open(output_path, "w") as outfile:
        json.dump(data, outfile, indent=4)

def run_alphafold(json_path):
    """
    AlphaFold3 を実行し、生成された各 summary_confidences.json 内の "ptm" 値の平均を返す関数。
    
    Parameters:
        json_dir (str): JSONファイルが存在するディレクトリ
        test_name (str): テスト名。ファイル名は {test_name}.json として利用され、また出力ディレクトリにも用いられる。
        
    Returns:
        float or None: 5 つの "ptm" 値の平均。値が取得できなかった場合は None を返す。
    """
    # json_path を json_dir と test_name から組み立てる
    
    # 作業開始前のカレントディレクトリを保存
    original_dir = os.getcwd()
    base_output_dir = os.path.join(OUTPUT_DIR, "output")
    print(f"{base_output_dir=}")
    try:
        # AlphaFold3 のディレクトリに移動
        os.chdir(ALPHAFOLD3DIR)
        
        # uv run で実行するコマンド引数をリスト形式で構築
        cmd = [
            "uv", "run", "run_alphafold.py",
            "--jackhmmer_binary_path", os.path.join(HMMER3_BINDIR, "jackhmmer"),
            "--nhmmer_binary_path", os.path.join(HMMER3_BINDIR, "nhmmer"),
            "--hmmalign_binary_path", os.path.join(HMMER3_BINDIR, "hmmalign"),
            "--hmmsearch_binary_path", os.path.join(HMMER3_BINDIR, "hmmsearch"),
            "--hmmbuild_binary_path", os.path.join(HMMER3_BINDIR, "hmmbuild"),
            "--db_dir", DB_DIR,
            "--model_dir", MODEL_DIR,
            "--json_path", json_path,
            "--jackhmmer_n_cpu", "32",
            "--output_dir", base_output_dir
        ]
        
        # AlphaFold3 の実行
        subprocess.run(cmd, check=True)
        print("AlphaFold3 の実行が正常に完了しました。")
        
    except subprocess.CalledProcessError as e:
        print("AlphaFold3 の実行中にエラーが発生しました:")
        print(e)
        return None
        
    finally:
        # 実行前のカレントディレクトリに戻す
        os.chdir(original_dir)
        print(f"カレントディレクトリを {original_dir} に戻しました。")
    
    # 出力ディレクトリのベースパスを指定（必要に応じて変更してください）
def get_iptm(test_name):
    base_output_dir = os.path.join(OUTPUT_DIR, "output")
    summary_file = os.path.join(base_output_dir, test_name, f"{test_name}_summary_confidences.json")
    try:
        with open(summary_file, "r") as f:
            summary = json.load(f)
            iptm = summary.get("iptm")
            if iptm is None:
                print(f"'{summary_file}' に 'iptm' キーが見つかりませんでした。")
                return None
            else:
                return iptm
    except Exception as e:
        print(f"{summary_file} の読み込み中にエラーが発生しました: {e}")
        return None

def get_loss(protein_string):
    """
    タンパク質の配列文字列を受け取り、3J0Aとの相互作用を予測し、
    ベイズ最適化用の損失値を返します。
    
    ベイズ最適化では最小化問題として扱うため、
    相互作用確率が高いほど損失値が低くなるよう変換します。
    
    Parameters:
    protein_string (str): タンパク質の配列文字列
    
    Returns:
    float: 最小化すべき損失値（相互作用確率の反転値: 1 - 確率）
    """
    if not hasattr(get_loss, 'counter'):
        get_loss.counter = 0
    get_loss.counter += 1
    test_name = f"mev_{get_loss.counter}"
    output_dir = f"{OUTPUT_DIR}/input_json"
    create_json_3J0A(protein_string, test_name, output_dir)
    run_alphafold(test_name, output_dir)
    interaction_prob = get_iptm(test_name)
    # 相互作用確率を取得
    #interaction_prob = run_interaction_prediction_from_string(protein_string)
    # 最小化問題として扱うため確率を反転（確率が高いほど損失が低くなる）
    loss = 1.0 - interaction_prob
    
    return loss

def get_losses(protein_strings):
    if not hasattr(get_losses, 'counter'):
        get_losses.counter = 0
    get_losses.counter += 1
    test_name = f"mevs_{get_losses.counter}"
    output_dir = f"{OUTPUT_DIR}/input_json"
    create_multiple_json_3J0A(protein_strings, test_name, output_dir)
    json_path = os.path.join(output_dir, f"{test_name}.json")
    run_alphafold(json_path)
    ret = [1.0 - get_iptm(f"{test_name}_{i}") for i in range(len(protein_strings))]
    return ret

def load_esm_model():
    model_name = "facebook/esm2_t33_650M_UR50D"  # 使用するESMモデル
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

# タンパク質の埋め込み表現を取得
def get_protein_embedding(protein, tokenizer, model):
    inputs = tokenizer(protein, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # 最終隠れ層の平均を埋め込み表現として使用
    embedding = outputs.last_hidden_state.mean(dim=1).numpy()
    return embedding.flatten()

# エピトープとリンカーを組み合わせてタンパク質を生成
def generate_protein(epitopes, linkers, order, linker_choices):
    """
    エピトープの順序とリンカーの選択に基づいてタンパク質配列を生成
    
    Parameters:
    epitopes (list): エピトープのリスト
    linkers (list): リンカーのリスト
    order (list): エピトープの順序を表すインデックスのリスト
    linker_choices (list): 使用するリンカーのインデックスのリスト
    
    Returns:
    str: 生成されたタンパク質配列
    """
    protein_parts = []
    protein_parts.append("MPSEKTFKQRRSFEQRVEDVRLIREQHPTKIPVIIERYKGEKQLPVLDKTKFLVPDHVNMSELIKIIRRRLQLNANQAFFLLVNGHSMVSVSTPISEVYESERDEDGFLYMVYASQETFGTAMAVEAAAK")
    for i, idx in enumerate(order):
        protein_parts.append(epitopes[idx])
        protein_parts.append(linkers[linker_choices[i]])
    protein_parts.append("HHHHHH")
    
    return "".join(protein_parts)

# ランダムな個体を生成
def create_random_individual(n_epitopes, n_linkers):
    """ランダムなエピトープ順序とリンカー選択を持つ個体を生成"""
    order = list(range(n_epitopes))
    random.shuffle(order)
    linker_choices = [random.randint(0, n_linkers-1) for _ in range(n_epitopes)]
    return (order, linker_choices)

# 初期集団の生成
def create_initial_population(n_epitopes, n_linkers, population_size=50):
    """ランダムな個体の集団を生成"""
    return [create_random_individual(n_epitopes, n_linkers) for _ in range(population_size)]

# 部分的マッピングクロスオーバー(PMX)
def pmx_crossover(parent1_order, parent2_order):
    """順序のクロスオーバー処理"""
    n = len(parent1_order)
    if n <= 2:
        return parent1_order.copy()
    
    # 交差点を2箇所選択
    cx_point1 = random.randint(0, n-2)
    cx_point2 = random.randint(cx_point1+1, n-1)
    
    # 子の初期化
    offspring = [-1] * n
    
    # 親1から交差点間のセグメントをコピー
    for i in range(cx_point1, cx_point2+1):
        offspring[i] = parent1_order[i]
    
    # マッピングを作成
    mapping = {}
    for i in range(cx_point1, cx_point2+1):
        if parent2_order[i] not in offspring:
            mapping[parent2_order[i]] = parent1_order[i]
    
    # 残りの位置を埋める
    for i in range(n):
        if i < cx_point1 or i > cx_point2:
            val = parent2_order[i]
            while val in mapping:
                val = mapping[val]
            if val not in offspring:
                offspring[i] = val
    
    # まだ埋まっていない位置に未使用の値を入れる
    unused = set(range(n)) - set(filter(lambda x: x != -1, offspring))
    for i in range(n):
        if offspring[i] == -1:
            offspring[i] = unused.pop()
    
    return offspring

# クロスオーバー: 2つの親から子を生成
def crossover(parent1, parent2):
    """2つの親から子を生成するクロスオーバー操作"""
    order1, linker_choices1 = parent1
    order2, linker_choices2 = parent2
    
    # 順序のクロスオーバーにPMXを使用
    child_order = pmx_crossover(order1, order2)
    
    # リンカー選択のクロスオーバー（一点交差）
    n_linkers = len(linker_choices1)
    if n_linkers > 0:
        cx_point = random.randint(0, n_linkers-1)
        child_linker_choices = linker_choices1[:cx_point] + linker_choices2[cx_point:]
    else:
        child_linker_choices = []
    
    return (child_order, child_linker_choices)

# 突然変異: ランダムな変更を導入
def mutate(individual, n_epitopes, n_linkers, mutation_rate=0.1):
    """個体に突然変異を導入"""
    order, linker_choices = individual
    order = order.copy()
    linker_choices = linker_choices.copy()
    
    # 順序の突然変異（スワップ）
    if random.random() < mutation_rate and len(order) > 1:
        idx1, idx2 = random.sample(range(len(order)), 2)
        order[idx1], order[idx2] = order[idx2], order[idx1]
    
    # リンカー選択の突然変異
    for i in range(len(linker_choices)):
        if random.random() < mutation_rate:
            linker_choices[i] = random.randint(0, n_linkers-1)
    
    return (order, linker_choices)

# 獲得関数: 期待改善(Expected Improvement)
def expected_improvement(mean, std, best_f, xi=0.01):
    """期待改善を計算する獲得関数"""
    with np.errstate(divide='ignore'):
        z = (mean - best_f - xi) / std
        ei = (mean - best_f - xi) * norm.cdf(z) + std * norm.pdf(z)
        ei[std == 0.0] = 0.0
        return ei

# 評価済みタンパク質のキャッシュ
protein_cache = {}

# タンパク質を評価（キャッシュを使用して再評価を防止）
def evaluate_protein(protein, tokenizer, model):
    """タンパク質の埋め込みと損失を取得（キャッシュ使用）"""
    if protein in protein_cache:
        return protein_cache[protein]['embedding'], protein_cache[protein]['fitness']
    
    embedding = get_protein_embedding(protein, tokenizer, model)
    fitness = get_loss(protein)
    
    protein_cache[protein] = {
        'embedding': embedding,
        'fitness': fitness
    }
    
    return embedding, fitness

def evaluate_multiple_proteins(proteins, tokenizer, model):
    """タンパク質の埋め込みと損失を取得（キャッシュ使用）"""
    queue_proteins = []
    for i, protein in enumerate(proteins):
        if not protein in protein_cache:
            queue_proteins.append(protein)
    losses = get_losses(queue_proteins)
    for i in range(len(queue_proteins)):
        protein = queue_proteins[i]
        fitness = losses[i]
        embedding = get_protein_embedding(protein, tokenizer, model)
        protein_cache[protein] = {
            'embedding': embedding,
            'fitness': fitness
        }

    embeddings = [protein_cache[protein]["embedding"] for protein in proteins]
    fitnesses = [protein_cache[protein]["fitness"] for protein in proteins]
    
    return embeddings, fitnesses
# トーナメント選択
def tournament_selection(population, fitnesses, tournament_size=3):
    """トーナメント選択により個体を選択"""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    return min(tournament_indices, key=lambda i: fitnesses[i])

# ベイズ最適化ガイド付き遺伝的アルゴリズムによるタンパク質最適化
def optimize_protein(epitopes, linkers, n_generations=1000, population_size=10, n_elite=5):
    """
    ベイズ最適化と遺伝的アルゴリズムを組み合わせたタンパク質最適化
    
    Parameters:
    epitopes (list): エピトープのリスト
    linkers (list): リンカーのリスト
    n_generations (int): 世代数
    population_size (int): 集団サイズ
    n_elite (int): エリート個体数
    
    Returns:
    tuple: 最適なタンパク質配列とその損失値
    """
    tokenizer, model = load_esm_model()
    
    # 初期集団を生成
    population = create_initial_population(len(epitopes), len(linkers), population_size)
    
    # 初期集団を評価
    proteins = [generate_protein(epitopes, linkers, order, linker_choices) for order, linker_choices in population]
    embeddings, fitnesses = evaluate_multiple_proteins(proteins, tokenizer, model)
    
    # 最良解の初期化
    best_idx = np.argmin(fitnesses)
    best_fitness = fitnesses[best_idx]
    best_individual = population[best_idx]
    best_protein = proteins[best_idx]
    
    print(f"初期ベスト損失: {best_fitness}")
    
    # ガウス過程の設定
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-6)
    
    # メイン最適化ループ
    for generation in range(n_generations):
        # ガウス過程を現在のデータに適合
        X = np.array(embeddings)
        y = np.array(fitnesses)
        gp.fit(X, y)
        
        # エリート個体の選択
        elite_indices = sorted(range(len(population)), key=lambda i: fitnesses[i])[:n_elite]
        next_population = [population[i] for i in elite_indices]
        
        # 残りの個体を生成
        candidates_count = 0
        while len(next_population) < population_size:
            candidates_count += 1
            if candidates_count > 1000:  # 無限ループ防止
                break
                
            # 選択方法の確率に基づく決定
            if random.random() < 0.7:  # 70%の確率で遺伝的操作、30%でランダム生成
                # トーナメント選択
                parent1_idx = tournament_selection(population, fitnesses)
                parent2_idx = tournament_selection(population, fitnesses)
                
                # クロスオーバーと突然変異
                child = crossover(population[parent1_idx], population[parent2_idx])
                child = mutate(child, len(epitopes), len(linkers))
            else:
                # ランダム生成（探索のため）
                child = create_random_individual(len(epitopes), len(linkers))
            
            # タンパク質を生成してチェック
            protein = generate_protein(epitopes, linkers, child[0], child[1])
            if protein in protein_cache:
                # すでに評価済みのタンパク質はスキップ
                continue
            
            # 埋め込みを取得してGPモデルで予測
            embedding = get_protein_embedding(protein, tokenizer, model)
            mean, std = gp.predict([embedding], return_std=True)
            
            # 期待改善を計算
            ei = expected_improvement(mean, std, best_fitness)
            
            # 有望な候補のみ追加
            if ei > 0.01 or random.random() < 0.1:  # EIが低くても10%の確率で追加
                next_population.append(child)
        
        # 新しい集団を評価
        population = next_population
        
        proteins = [generate_protein(epitopes, linkers, order, linker_choices) for order, linker_choices in population]
        embeddings, fitnesses = evaluate_multiple_proteins(proteins, tokenizer, model)
        
        # 最良個体の更新
        current_best_idx = np.argmin(fitnesses)
        current_best_fitness = fitnesses[current_best_idx]
        
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[current_best_idx]
            best_protein = proteins[current_best_idx]
            logging.info(f"世代 {generation+1}/{n_generations}, 新しいベスト損失: {best_fitness}, {best_protein=}")
        else:
            logging.info(f"世代 {generation+1}/{n_generations}, ベスト損失: {best_fitness}, {best_protein=}")
    
    return best_protein, best_fitness

# 使用例
def main():
    global OUTPUT_DIR
    OUTPUT_DIR = get_checkpoint_dir("checkpoints", "bayes")
    print(f"{OUTPUT_DIR=}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    shutil.copy(__file__, os.path.join(OUTPUT_DIR, os.path.basename(__file__)))
    
    # ロギングの設定
    log_file_path = os.path.join(OUTPUT_DIR, "output.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    # エピトープとリンカーの例
    epitopes = [
        'LIKKNDAYPTIKISYNNTNREDL',
        'MENIVLLLAIVSLVKS',
        'TIMEKNV',
        'EEISGVKLESVGTYQILSIYSTAASSLALAIMMAGLSLWMCSNGSLQCRICI',
        'VLMENERTL',
        'RMDFFWTILK',
        'LYDKVRLQL',
        'DTIMEKNVTV',
        'SVRNGTYDY',
        'REEISGVKL',
        'TIMEKNVTV',
        'RLKREEISGV',
        'MPFHNIHPL',
        'NLYDKVRLQL',
        'FHNIHPLTI',
        'EWSYIVERA',
        'WSYIVERAN',
        'IKISYNNTN',
        'MEKNVTVTH',
        'FHDSNVKNL',
        'YISVGTSTL',
        'LSIYSTAAS',
        'IDKMNTQFE',
        'ISYNNTNRE',
        'IENLNKKME'
    ]
    
    linkers = ["AAY", "GPGPG", "KK"]
    
    best_protein, best_loss = optimize_protein(epitopes, linkers)
    print(f"最適なタンパク質: {best_protein}")
    print(f"最適な損失値: {best_loss}")


if __name__ == "__main__":
    main()
    #test_name = "test"
    #output_dir = "/home/hikari/alphafold3/input_json"
    ##create_json_3J0A("MPSEKTFKQRRSFEQRVEDVRLIREQHPTKIPVIIERYKGEKQLPVLDKTKFLVPDHVNMSELIKIIRRRLQLNANQAFFLLVNGHSMVSVSTPISEVYESERDEDGFLYMVYASQETFGTAMAVEAAAK", test_name, output_dir)
    #create_json("HHHHHH","FHDSNVKNL",test_name, output_dir)
    #run_alphafold(test_name, output_dir)