from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from stitch_utils import as_rgb_float, harris_sift_points_desc, parse_list

# steps
# 1) load pairwise descriptor matrices from step 3
# 2) sweep thresholds and top-k values for both correlation and euclidean metrics
# 3) plot match counts vs parameter values
# 4) select one final pair set and save for ransac


def unique_one_to_one(pairs, scores, higher_is_better):
	# many-to-one matches are noisy for geometry and inflate match counts.
	# rank candidates by score, then greedily keep only unused descriptor indices from both sides.
	if len(pairs) == 0:
		return np.empty((0, 2), dtype=int), np.empty((0,), dtype=np.float32)

	order = np.argsort(-scores if higher_is_better else scores)
	used_i = set()
	used_j = set()
	kept_pairs = []
	kept_scores = []

	for idx in order:
		i, j = int(pairs[idx, 0]), int(pairs[idx, 1])
		if i in used_i or j in used_j:
			continue
		used_i.add(i)
		used_j.add(j)
		kept_pairs.append([i, j])
		kept_scores.append(float(scores[idx]))

	return np.asarray(kept_pairs, dtype=int), np.asarray(kept_scores, dtype=np.float32)


def topk_pairs(matrix, k, higher_is_better):
	# flatten matrix, take top-k indices, convert back to descriptor index pairs.
	if matrix.size == 0:
		return np.empty((0, 2), dtype=int), np.empty((0,), dtype=np.float32)

	flat = matrix.reshape(-1)
	k = max(1, min(int(k), len(flat)))
	# argpartition gives the k smallest/largest indices without fully sorting the array.
	idx = np.argpartition(-flat if higher_is_better else flat, kth=k - 1)[:k]
	i, j = np.unravel_index(idx, matrix.shape)
	pairs = np.column_stack([i, j]).astype(int)
	scores = matrix[i, j].astype(np.float32)
	return unique_one_to_one(pairs, scores, higher_is_better)


def threshold_pairs(matrix, thr, higher_is_better):
	# keep matrix entries that pass threshold and then enforce one-to-one uniqueness.
	if matrix.size == 0:
		return np.empty((0, 2), dtype=int), np.empty((0,), dtype=np.float32)

	if higher_is_better:
		i, j = np.where(matrix >= float(thr))
	else:
		i, j = np.where(matrix <= float(thr))
	pairs = np.column_stack([i, j]).astype(int)
	scores = matrix[i, j].astype(np.float32)
	return unique_one_to_one(pairs, scores, higher_is_better)


def draw_matches(img1, p1, img2, p2, pairs, out_path, title, max_draw):
	# place images side-by-side on one canvas and draw lines for selected matches.
	v1 = as_rgb_float(img1)
	v2 = as_rgb_float(img2)
	h1, w1 = v1.shape[:2]
	h2, w2 = v2.shape[:2]
	canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.float32)
	canvas[:h1, :w1] = v1
	canvas[:h2, w1 : w1 + w2] = v2

	fig, ax = plt.subplots(1, 1, figsize=(14, 6))
	ax.imshow(canvas)
	for i, j in pairs[: max(1, int(max_draw))]:
		y1, x1 = p1[i]
		y2, x2 = p2[j]
		ax.plot([x1, x2 + w1], [y1, y2], color="yellow", lw=0.7, alpha=0.6)
	ax.set_title(f"{title} | matches={len(pairs)}")
	ax.axis("off")
	fig.tight_layout()
	fig.savefig(out_path, dpi=150)
	plt.close(fig)


def main():
	p = argparse.ArgumentParser(description="step 4: match selection sensitivity")
	p.add_argument("--img1", default="mnt1.JPG")
	p.add_argument("--img2", default="mnt2.JPG")
	p.add_argument("--sigma", type=float, default=1.5)
	p.add_argument("--k", type=float, default=0.05)
	p.add_argument("--threshold_rel", type=float, default=0.01)
	p.add_argument("--min_distance", type=int, default=5)
	p.add_argument("--max_points", type=int, default=1000)
	p.add_argument("--sift_keypoint_size", type=float, default=8.0)

	p.add_argument("--corr_npy", default="outputs/step3_norm_correlation.npy")
	p.add_argument("--euc_npy", default="outputs/step3_norm_euclidean.npy")

	p.add_argument("--corr_thresholds", default="0.50,0.60,0.70,0.80,0.90")
	p.add_argument("--euc_thresholds", default="0.50,0.70,0.90,1.10,1.30")
	p.add_argument("--topk_values", default="50,100,200,400")

	p.add_argument("--final_metric", choices=["corr", "euc"], default="corr")
	p.add_argument("--final_mode", choices=["topk", "threshold"], default="topk")
	p.add_argument("--final_value", type=float, default=200)

	p.add_argument("--out_pairs", default="outputs/step4_corr_topk_pairs.npy")
	p.add_argument("--out_plot", default="outputs/step4_sensitivity_plots.png")
	p.add_argument("--out_matches", default="outputs/step4_selected_matches.png")
	p.add_argument("--max_draw_matches", type=int, default=200)
	a = p.parse_args()

	corr = np.load(a.corr_npy)
	euc = np.load(a.euc_npy)

	img1 = io.imread(a.img1)
	img2 = io.imread(a.img2)
	p1, _ = harris_sift_points_desc(img1, a.sigma, a.k, a.threshold_rel, a.min_distance, a.max_points, a.sift_keypoint_size)
	p2, _ = harris_sift_points_desc(img2, a.sigma, a.k, a.threshold_rel, a.min_distance, a.max_points, a.sift_keypoint_size)

	corr_th = parse_list(a.corr_thresholds, float)
	euc_th = parse_list(a.euc_thresholds, float)
	topk_vals = parse_list(a.topk_values, int)

	# for the sensitivity plot use raw counts (no uniqueness filter) so the plot
	# shows the true effect of each parameter rather than hitting a uniqueness ceiling.
	corr_th_counts = [int(np.sum(corr >= v)) for v in corr_th]
	euc_th_counts  = [int(np.sum(euc  <= v)) for v in euc_th]
	corr_k_counts  = [min(int(v), corr.size) for v in topk_vals]
	euc_k_counts   = [min(int(v), euc.size)  for v in topk_vals]

	# choose one final pair set for step 5.
	# hib = higher_is_better: true for correlation (higher = more similar), false for euclidean.
	if a.final_metric == "corr":
		mat = corr
		hib = True
	else:
		mat = euc
		hib = False

	if a.final_mode == "topk":
		final_pairs, final_scores = topk_pairs(mat, int(a.final_value), hib)
	else:
		final_pairs, final_scores = threshold_pairs(mat, float(a.final_value), hib)

	for f in [a.out_pairs, a.out_plot, a.out_matches]:
		Path(f).parent.mkdir(parents=True, exist_ok=True)

	np.save(a.out_pairs, final_pairs)
	draw_matches(
		img1,
		p1,
		img2,
		p2,
		final_pairs,
		a.out_matches,
		title=f"{a.final_metric} {a.final_mode} value={a.final_value}",
		max_draw=a.max_draw_matches,
	)

	fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
	ax[0].plot(corr_th, corr_th_counts, marker="o", label="corr threshold")
	ax[0].plot(euc_th, euc_th_counts, marker="o", label="euc threshold")
	ax[0].set_xlabel("threshold")
	ax[0].set_ylabel("selected matches")
	ax[0].set_title("threshold sensitivity")
	ax[0].grid(alpha=0.3)
	ax[0].legend()

	ax[1].plot(topk_vals, corr_k_counts, marker="o", label="corr top-k")
	ax[1].plot(topk_vals, euc_k_counts, marker="o", label="euc top-k")
	ax[1].set_xlabel("k")
	ax[1].set_ylabel("selected matches")
	ax[1].set_title("top-k sensitivity")
	ax[1].grid(alpha=0.3)
	ax[1].legend()

	fig.suptitle("step 4: match selection sensitivity")
	fig.tight_layout()
	fig.savefig(a.out_plot, dpi=150)
	plt.close(fig)

	print(f"saved selected pairs: {a.out_pairs}")
	print(f"saved selected matches visualization: {a.out_matches}")
	print(f"saved sensitivity plot: {a.out_plot}")
	print(f"selected pair count for step 5: {len(final_pairs)}")
	print(f"selected score median: {float(np.median(final_scores)):.4f}")


if __name__ == "__main__":
	main()
