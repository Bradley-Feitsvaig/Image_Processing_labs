# Fares Fares, 311136287
# Bradley Feitsvaig, 311183073

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import os
import shutil
import sys


# matches is of (3|4 X 2 X 2) size. Each row is a match - pair of (kp1,kp2) where kpi = (x,y)
def get_transform(matches, is_affine):
    src_points, dst_points = matches[:, 0], matches[:, 1]
    # Add your code here
    if is_affine:
        # Compute the affine transformation matrix
        T = cv2.getAffineTransform(np.float32(src_points), np.float32(dst_points))
    else:
        # Compute the projective (homography) transformation matrix
        T = cv2.getPerspectiveTransform(np.float32(src_points), np.float32(dst_points))
    return T


def stitch(img1, img2):
    return cv2.max(img1, img2)


# Output size is (w,h)
def inverse_transform_target_image(target_img, original_transform, output_size):
    # Add your code here
    is_affine = original_transform.shape[0] == 2

    if is_affine:
        # Invert the affine transformation matrix
        T_inv = cv2.invertAffineTransform(original_transform)
        # Apply the inverse affine transformation
        transformed_img = cv2.warpAffine(target_img, T_inv, dsize=output_size, flags=cv2.INTER_LINEAR)
    else:
        # Invert the projective (homography) transformation matrix
        T_inv = np.linalg.inv(original_transform)
        # Apply the inverse projective transformation
        transformed_img = cv2.warpPerspective(target_img, T_inv, dsize=output_size, flags=cv2.INTER_LINEAR)

    return transformed_img


# returns list of pieces file names
def prepare_puzzle(puzzle_dir):
    # Path for the edited pieces directory
    edited = os.path.join(puzzle_dir, 'abs_pieces')

    # Create or recreate the edited pieces directory
    if os.path.exists(edited):
        shutil.rmtree(edited)
    os.mkdir(edited)

    # Determine if the puzzle is affine based on the directory name
    affine = 4 - int("affine" in puzzle_dir)

    # Load matches data
    matches_data = os.path.join(puzzle_dir, 'matches.txt')
    # Preprocessing of matches.txt to be done outside this function, assuming it's ready to load
    n_images = len(os.listdir(os.path.join(puzzle_dir, 'pieces')))
    matches = np.loadtxt(matches_data, dtype=np.int64).reshape(n_images - 1, affine, 2, 2)

    return matches, affine == 3, n_images


if __name__ == '__main__':
    lst = ['puzzle_affine_1', 'puzzle_affine_2', 'puzzle_homography_1']
    # lst = ['puzzle_homography_1']

    for puzzle_dir in lst:
        print(f'Starting {puzzle_dir}')

        puzzle = os.path.join('puzzles', puzzle_dir)

        pieces_pth = os.path.join(puzzle, 'pieces')
        edited = os.path.join(puzzle, 'abs_pieces')

        matches, is_affine, n_images = prepare_puzzle(puzzle)
        first_piece = cv2.imread(f'{os.path.join(pieces_pth)}/piece_1.jpg')
        cv2.imwrite((f'{os.path.join(edited)}/piece_{1}_absolute.jpg'), first_piece)
        final_puzzle = first_piece
        for i in range(2, n_images+1):
            img = cv2.imread(f'{os.path.join(pieces_pth)}/piece_{i}.jpg')
            T = get_transform(matches[i-2], is_affine)
            transformed_img = inverse_transform_target_image(img, T, (first_piece.shape[1], first_piece.shape[0]))
            cv2.imwrite((f'{os.path.join(edited)}/piece_{i}_absolute.jpg'), transformed_img)
            final_puzzle = stitch(final_puzzle,transformed_img)
        sol_file = f'solution.jpg'
        cv2.imwrite(os.path.join(puzzle, sol_file), final_puzzle)
