"""Data cleaning and visualisation for messy_dataset.csv."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from word2number import w2n
    HAS_W2N = True
except ImportError:
    HAS_W2N = False
    print("Note: 'word2number' not installed — text numbers will become NaN.")
    print("Install with: pip install word2number\n")

# Extend as new non-standard country values are encountered
COUNTRY_MAP = {
    'AU':          'AUS',
    'Australia':   'AUS',
    'New Zealand': 'NZL',
}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def first_non_null(series):
    """Return the first non-null value in a Series, or NaN if all null."""
    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) > 0 else np.nan


def to_numeric_smart(val):
    """Try float conversion first; fall back to word-to-number."""
    if pd.isna(val):
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        if HAS_W2N:
            try:
                return float(w2n.word_to_num(str(val)))
            except ValueError:
                return np.nan
        return np.nan


def is_non_numeric_string(x):
    """Return True if x is a string that cannot be parsed as a float."""
    if pd.isna(x):
        return False
    try:
        float(x)
        return False
    except (ValueError, TypeError):
        return True


def add_notes(fig, lines, fontsize=7.5, y=0.01, pad=0.5):
    """Render a cleaning-notes box at the bottom of the figure."""
    fig.text(
        0.01, y, "\n".join(lines),
        fontsize=fontsize, va='bottom', fontfamily='monospace',
        bbox={
            'boxstyle': f'round,pad={pad}',
            'facecolor': '#fffde7',
            'alpha': 0.85,
        }
    )


# ── LOADING ───────────────────────────────────────────────────────────────────

def load_data(path):
    """Load CSV as strings and normalise blank cells to NaN."""
    df = pd.read_csv(path, dtype=str)
    df = df.replace(r'^\s*$', np.nan, regex=True)
    return df


# ── CLEANING ──────────────────────────────────────────────────────────────────

def clean_data(df):
    """Apply all cleaning steps; return (cleaned_df, cleaning_log)."""
    df = df.copy()
    log = []

    # -- Merge duplicate IDs --------------------------------------------------
    dup_ids = df[
        df['ID'].notna() & df.duplicated(subset=['ID'], keep=False)
    ]['ID'].unique()
    if len(dup_ids) > 0:
        df_with_id = df[df['ID'].notna()].groupby(
            'ID', as_index=False).agg(first_non_null)
        df_no_id = df[df['ID'].isna()]
        df = pd.concat([df_with_id, df_no_id], ignore_index=True)
        log.append(
            f"Duplicate IDs: {len(dup_ids)} ID(s) had multiple rows "
            f"→ merged (first non-null value per column retained)"
        )

    # -- Fill missing IDs from sequential gap ---------------------------------
    numeric_ids = pd.to_numeric(df['ID'], errors='coerce').dropna().astype(int)
    if len(numeric_ids) > 0:
        gaps = sorted(
            set(range(numeric_ids.min(), numeric_ids.max() + 1)) - set(numeric_ids)
        )
        missing_mask = df['ID'].isna()
        n_missing = int(missing_mask.sum())
        if n_missing > 0 and len(gaps) >= n_missing:
            df.loc[missing_mask, 'ID'] = [str(g) for g in gaps[:n_missing]]
            log.append(
                f"Missing ID: {n_missing} row(s) assigned inferred "
                f"sequential value(s) {gaps[:n_missing]}"
            )
        elif n_missing > 0:
            log.append(
                f"Missing ID: {n_missing} row(s) — "
                f"gaps insufficient to infer, left as NaN"
            )

    # -- Standardise Country --------------------------------------------------
    original_country = df['Country'].copy()
    df['Country'] = df['Country'].replace(COUNTRY_MAP)
    n_standardised = int((df['Country'] != original_country).sum())
    if n_standardised > 0:
        log.append(
            f"Country: {n_standardised} value(s) standardised via mapping "
            f"(e.g. 'AU' → 'AUS')"
        )

    # -- Convert text → numeric (Age and Salary) ------------------------------
    for col in ['Age', 'Salary']:
        original = df[col].copy()
        df[col] = df[col].apply(to_numeric_smart)
        n_converted = int(
            (original.apply(is_non_numeric_string) & df[col].notna()).sum()
        )
        n_failed = int((original.notna() & df[col].isna()).sum())
        entry = f"{col}: {n_converted} text value(s) converted to numeric"
        if n_failed > 0:
            entry += f", {n_failed} unparseable → NaN"
        log.append(entry)

    # -- Convert remaining types ----------------------------------------------
    df['ID'] = pd.to_numeric(df['ID'], errors='coerce')

    original_date = df['Join Date'].copy()
    df['Join Date'] = pd.to_datetime(
        df['Join Date'], dayfirst=True, errors='coerce')
    n_invalid = int((original_date.notna() & df['Join Date'].isna()).sum())
    n_missing_date = int(original_date.isna().sum())
    log.append(
        f"Join Date: {n_invalid} invalid value(s) → NaT, "
        f"{n_missing_date} already missing → NaT"
    )

    return df, log


# ── VISUALISATION ─────────────────────────────────────────────────────────────

def prepare_box(df):
    """Drop rows missing Salary or Country; build cleaning notes."""
    n_drop_salary = int(df['Salary'].isna().sum())
    n_drop_country = int(df['Country'].isna().sum())
    df_plot = df.dropna(subset=['Salary', 'Country'])

    notes = ["Data Cleaning:", "  DROPPED:"]
    if n_drop_salary > 0:
        notes.append(f"    • {n_drop_salary} row(s): missing Salary")
    if n_drop_country > 0:
        notes.append(f"    • {n_drop_country} row(s): missing Country")
    return df_plot, notes


def plot_box(df_plot, notes):
    """Box plot of Salary by Country."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(bottom=0.22)
    sns.boxplot(data=df_plot, x='Country', y='Salary', ax=ax, palette='Set2')
    ax.set_title('Salary Distribution by Country — Box Plot')
    ax.set_xlabel('Country')
    ax.set_ylabel('Salary')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))
    add_notes(fig, notes)
    plt.savefig('boxplot.png', bbox_inches='tight', dpi=150)
    plt.show()
    print("Saved: boxplot.png")


def plot_age_salary(df):
    """Scatter plot of Age vs Salary with a regression line."""
    df_plot = df.dropna(subset=['Age', 'Salary'])
    n_dropped = int(df['Age'].isna().sum() + df['Salary'].isna().sum())

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(bottom=0.22)
    sns.regplot(data=df_plot, x='Age', y='Salary', ax=ax,
                scatter_kws={'alpha': 0.7}, line_kws={'color': 'red'})
    ax.set_title('Age vs Salary — Scatter Plot with Regression Line')
    ax.set_xlabel('Age')
    ax.set_ylabel('Salary')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))

    notes = ["Data Cleaning:", "  DROPPED:"]
    if n_dropped > 0:
        notes.append(f"    • {n_dropped} row(s): missing Age or Salary")
    add_notes(fig, notes)
    plt.savefig('age_salary.png', bbox_inches='tight', dpi=150)
    plt.show()
    print("Saved: age_salary.png")


def plot_heatmap(df):
    """Correlation heatmap of Age, Salary, and Years of Service."""
    df_hm = df.copy()
    today = pd.Timestamp.today().normalize()
    df_hm['Years of Service'] = (
        (today - df_hm['Join Date']).dt.days / 365.25
    ).round(1)

    n_no_yos = int(df_hm['Years of Service'].isna().sum())
    n_drop_age_sal = int((df_hm['Age'].isna() | df_hm['Salary'].isna()).sum())
    df_hm = df_hm.dropna(subset=['Age', 'Salary'])

    notes = ["Data Cleaning:", "  DROPPED / PARTIAL:"]
    if n_drop_age_sal > 0:
        notes.append(
            f"    • {n_drop_age_sal} row(s): missing Age or Salary → excluded"
        )
    if n_no_yos > 0:
        notes.append(
            f"    • {n_no_yos} row(s): invalid/missing Join Date "
            f"→ 'Years of Service' is NaN (corr. uses available data only)"
        )

    corr = df_hm[['Age', 'Salary', 'Years of Service']].corr(method='pearson')

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.subplots_adjust(bottom=0.28)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                ax=ax, vmin=-1, vmax=1, square=True, linewidths=0.5)
    ax.set_title('Pearson Correlation Heatmap — Age, Salary, Years of Service')
    add_notes(fig, notes)
    plt.savefig('heatmap.png', bbox_inches='tight', dpi=150)
    plt.show()
    print("Saved: heatmap.png")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    """Load, clean, and visualise the dataset."""
    # Load
    df_raw = load_data('messy_dataset.csv')
    print("=" * 65)
    print("RAW DATA")
    print("=" * 65)
    print(df_raw.to_string(index=False))
    print()

    # Clean
    df, log = clean_data(df_raw)
    print("=" * 65)
    print("CLEANING SUMMARY")
    print("=" * 65)
    for entry in log:
        print(f"  • {entry}")
    print()
    print("CLEANED DATA:")
    print(df.to_string(index=False))
    print()

    # Visualise
    df_box, box_notes = prepare_box(df)
    plot_box(df_box, box_notes)
    plot_age_salary(df)
    plot_heatmap(df)


if __name__ == '__main__':
    main()
