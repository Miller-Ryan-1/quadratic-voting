# Quadratic Voting Project

## Overview

This project introduces a method to calculate the optimal number of Quadratic Voting (QV) credits (C) to issue to voters in a QV vote, based on the number of distinct items (N) they are voting on. The goal is to identify the most general form of  𝐶𝑁=𝑓(𝑁), i.e., given a quadratic ballot with N choices, how many voting credits should each voter receive?

## Files

This repository contains:

1. The primary Jupyter notebook: `qv_voting_credits.ipynb`.
2. An older draft notebook: `qv_optimal_dc_dv.ipynb`.
3. A Python file with standalone functions developed in the process of creating the notebooks: `qv_1.py`.

## Usage

You can clone this repository and run the notebooks on your local machine using standard Git operations. Additionally, you can run the primary notebook in your browser via Google Colab. Here is the link to the Colab notebook: [qv_voting_credits.ipynb](https://colab.research.google.com/drive/1t_mypRLKpeYCAkg13p3IdUltno6rNOSB?usp=sharing).

## Results

The project begins by defining the characteristics of a QV vote: 

1. Paid Votes vs Standard Issue vs Mix
2. Discrete vs Continuous Votes
3. Discrete vs Continuous Voting Credits
4. One-time vs Multiple Ballots
5. Priority ('Positive-Only') vs Polar Voting
6. Individual vs Collective Agency

After initially focusing on discrete credit, discrete vote quadratic votes (explored in the 'qv_optimal_dc_dv.ipynb' notebook), it was determined that it wasn't an effective functional way of voting. The focus then shifted to discrete credit continuous vote QV with polar voting, and priority voting was explored concurrently.

No clear computational patterns emerged to support any particular credit issuance, though there is still much to investigate. Analytically, it supports a credit issuance of C = (N+3)//4 * N for polar, discrete credit continuous vote. For priority voting, it equals (N//2) * N.

## License

"This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details"