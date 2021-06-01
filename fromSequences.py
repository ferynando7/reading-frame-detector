import click
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

@click.command()
@click.option('--input', '-i', default='data/sequences.csv')
@click.option('--output', '-o', default='data/fasta.faa')


def main(input, output):
    df = pd.read_csv(input, header=None, usecols=[1])
    logging.info(df)


    out_file = open(output, 'w')

    count = 0
    for idx, _ in df.iterrows():
        out_file.write(f">{count}\n{df.iloc[idx, 0]}\n")


main()