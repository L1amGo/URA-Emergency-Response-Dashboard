import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

st.set_page_config(page_title="Emergency Response Dashboard", layout="wide")

# --------------------
# File upload + guard
# --------------------
uploaded_file = st.file_uploader("Upload masterlogURA.xlsx", type=["xlsx"])
if not uploaded_file:
    st.info("Please upload the Excel file to begin.")
    st.stop()

# ======================
# LOAD & PREPROCESS DATA
# ======================
df = pd.read_excel(uploaded_file)

# Start Date parsing
if "Start Date (mm-dd-yyyy)" in df.columns:
    df["Start Date"] = pd.to_datetime(df["Start Date (mm-dd-yyyy)"], errors="coerce")
else:
    df["Start Date"] = pd.to_datetime(df.get("Start Date"), errors="coerce")

# Month name column if missing
if "Month" not in df.columns and "Start Date" in df.columns:
    df["Month"] = df["Start Date"].dt.month_name()

def month_order():
    return [pd.Timestamp(month=i, day=1, year=2000).strftime('%B') for i in range(1, 13)]

month_options = month_order()

def create_bins(date, bin_size):
    """Return (label string, bin start date) for a Start Date."""
    if pd.isna(date):
        return None, None
    start_month = ((date.month - 1) // bin_size) * bin_size + 1
    end_month = start_month + bin_size - 1
    year = date.year
    if end_month > 12:
        end_month -= 12
        year += 1
    start_label = pd.Timestamp(year=date.year, month=start_month, day=1)
    label_str = (
        f"{start_label.strftime('%b')} {start_label.year}"
        if bin_size == 1
        else f"{start_label.strftime('%b')}-"
             f"{(start_label + pd.DateOffset(months=bin_size-1)).strftime('%b')} "
             f"{start_label.year}"
    )
    return label_str, start_label

# ============
# TOP-LEVEL TABS
# ============
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Basic Statistics",
        "Events Over Time – Specific",
        "Events Over Time – Top 10",
        "Seasonality Analysis",
    ]
)

# =========================
# TAB 1: BASIC STATISTICS
# =========================
with tab1:
    st.header("Basic Statistics")

    filtered_basic = df.copy()

    col1, col2 = st.columns([1, 1])

    # Top 10 Emergency Categories
    if "Emergency Category" in filtered_basic.columns:
        top_cat = (
            filtered_basic["Emergency Category"]
            .value_counts()
            .nlargest(10)
            .reset_index()
        )
        top_cat.columns = ["Emergency Category", "Count"]
    else:
        top_cat = pd.DataFrame(columns=["Emergency Category", "Count"])

    chart_cat = alt.Chart(top_cat).mark_bar(color="#38598b").encode(
        y=alt.Y(
            "Emergency Category:N",
            sort="-x",
            axis=alt.Axis(titleFontSize=12, labelFontSize=12, title=None)
        ),
        x=alt.X("Count:Q"),
        tooltip=["Emergency Category", "Count"]
    ).properties(
        height=330,
        width=380,
        padding=0,
        title="Top 10 Emergency Categories"
    )

    # Hazards by Month (Total)
    if "Month" in filtered_basic.columns:
        month_bar_data = (
            filtered_basic
            .groupby("Month")
            .size()
            .reindex(month_options, fill_value=0)
            .reset_index()
        )
        month_bar_data.columns = ["Month", "Count"]
    else:
        month_bar_data = pd.DataFrame(
            {"Month": month_options, "Count": [0] * 12}
        )

    hazards_by_month = alt.Chart(month_bar_data).mark_bar(color="#2b83ba").encode(
        x=alt.X('Month:N', sort=month_options, axis=alt.Axis(title=None)),
        y=alt.Y('Count:Q', axis=alt.Axis(title=None)),
        tooltip=['Month', 'Count']
    ).properties(
        height=330,
        width=380,
        padding=0,
        title="Hazards by Month (Total)"
    )

    with col1:
        st.altair_chart(chart_cat, use_container_width=True)
    with col2:
        st.altair_chart(hazards_by_month, use_container_width=True)

    # Metrics and mini hazard charts by year
    subcols = st.columns(6)
    with subcols[0]:
        st.markdown("**Total People Assisted**")
        total_people = int(filtered_basic.get("# of people assisted",
                                              pd.Series(dtype=float)).sum())
        st.metric("Total", total_people)

    if "Emergency Category" in filtered_basic.columns:
        hazard_total = (
            filtered_basic["Emergency Category"]
            .value_counts()
            .nlargest(5)
            .reset_index()
        )
        hazard_total.columns = ["Emergency Category", "Total Count"]
    else:
        hazard_total = pd.DataFrame(columns=["Emergency Category", "Total Count"])

    for i, row in hazard_total.iterrows():
        hazard = row["Emergency Category"]
        hazard_events = filtered_basic[
            filtered_basic["Emergency Category"] == hazard
        ].copy()
        with subcols[i + 1]:
            st.markdown(f"**{hazard}**")
            st.markdown(f"Total: {row['Total Count']}")
            if "Start Date" in hazard_events.columns:
                hazard_events = hazard_events.dropna(subset=["Start Date"])
                hazard_events["Year"] = hazard_events["Start Date"].dt.year
                yearly_counts = (
                    hazard_events
                    .groupby("Year")
                    .size()
                    .reset_index(name="EventCount")
                )
                if not yearly_counts.empty:
                    bar_chart = alt.Chart(yearly_counts).mark_bar(color="#2b83ba").encode(
                        x=alt.X('Year:O', axis=alt.Axis(title=None, labelAngle=0)),
                        y=alt.Y('EventCount:Q', axis=alt.Axis(title=None)),
                        tooltip=['Year', 'EventCount']
                    ).properties(height=190, width=190)
                    st.altair_chart(bar_chart, use_container_width=True)
                else:
                    st.warning("No date data.")
            else:
                st.warning("No date data.")

# =========================
# BIN-BASED FILTER HELPERS
# =========================
def bin_filters(bin_size_key_prefix: str):
    bin_options = [3, 4, 6, 12]
    bin_labels_text = {
        3: "Quarterly (3 months)",
        4: "Every 4 months",
        6: "Semi-annually (6 months)",
        12: "Annually (12 months)",
    }

    bin_size = st.selectbox(
        "Bin size",
        options=bin_options,
        index=2,  # default to 6 months
        key=f"{bin_size_key_prefix}_binsize",
        format_func=lambda v: bin_labels_text.get(v, f"{v} months"),
    )

    bins_df = df.copy()
    bins_all = bins_df["Start Date"].apply(lambda d: create_bins(d, bin_size))
    bins_df["Bin Label"] = bins_all.apply(lambda x: x[0])
    bins_df["Bin Date"] = bins_all.apply(lambda x: x[1])
    bins_df = bins_df.dropna(subset=["Bin Label", "Bin Date"])

    if bins_df.empty:
        return bin_size, None, None, bins_df

    bin_dates_sorted = sorted(bins_df["Bin Date"].unique())
    bin_labels_sorted = [
        bins_df[bins_df["Bin Date"] == b]["Bin Label"].iloc[0]
        for b in bin_dates_sorted
    ]

    start_label = st.selectbox(
        "Start period",
        options=bin_labels_sorted,
        index=0,
        key=f"{bin_size_key_prefix}_start",
    )
    end_label = st.selectbox(
        "End period",
        options=bin_labels_sorted,
        index=len(bin_labels_sorted) - 1,
        key=f"{bin_size_key_prefix}_end",
    )

    label_to_date = {lbl: dt for lbl, dt in zip(bin_labels_sorted, bin_dates_sorted)}
    start_bin_date = label_to_date[start_label]
    end_bin_date = label_to_date[end_label]

    return bin_size, start_bin_date, end_bin_date, bins_df

def apply_bin_range_filter(bins_df, start_date, end_date):
    if bins_df is None or start_date is None or end_date is None or bins_df.empty:
        return pd.DataFrame(columns=bins_df.columns if bins_df is not None else [])
    return bins_df[
        (bins_df["Bin Date"] >= start_date) & (bins_df["Bin Date"] <= end_date)
    ].copy()

def get_full_bin_data(base_df, hazard):
    if base_df.empty:
        return pd.DataFrame(columns=["Bin Date", "Bin Label", "Count"])

    bin_dates_sorted = sorted(base_df["Bin Date"].unique())
    bin_labels_sorted = [
        base_df[base_df["Bin Date"] == b]["Bin Label"].iloc[0]
        for b in bin_dates_sorted
    ]

    filtered_bins = base_df[base_df["Emergency Category"] == hazard]

    if filtered_bins.empty:
        all_bins = pd.DataFrame(
            {"Bin Date": bin_dates_sorted, "Bin Label": bin_labels_sorted}
        )
        all_bins["Count"] = 0
        return all_bins

    counts = (
        filtered_bins
        .groupby(["Bin Date", "Bin Label"])
        .size()
        .reset_index(name="Count")
    )

    all_bins = pd.DataFrame(
        {"Bin Date": bin_dates_sorted, "Bin Label": bin_labels_sorted}
    )
    merged = pd.merge(all_bins, counts, how="left", on=["Bin Date", "Bin Label"])
    merged["Count"] = merged["Count"].fillna(0)
    return merged.sort_values("Bin Date")

# =========================
# TAB 2: MAIN HAZARD
# =========================
with tab2:
    st.header("Events Over Time – Main Hazard")

    left2, right2 = st.columns([1, 4])

    with left2:
        bin_size_2, start_bin_2, end_bin_2, bins_df_2 = bin_filters("Main_Hazard")

        key_suffix = "Main_Hazard"
        if "Emergency Category" in df.columns:
            categories_all_2 = sorted(df["Emergency Category"].dropna().unique())
            if not df["Emergency Category"].dropna().empty:
                default_top = df["Emergency Category"].value_counts().idxmax()
                default_index = categories_all_2.index(default_top)
            else:
                default_index = 0
        else:
            categories_all_2 = []
            default_index = 0

        selected_cat_2 = st.selectbox(
            "Emergency Category",
            options=categories_all_2,
            index=default_index if categories_all_2 else 0,
            key=f"category_{key_suffix}",
        ) if categories_all_2 else None

    with right2:
        df_trend_2 = apply_bin_range_filter(bins_df_2, start_bin_2, end_bin_2)

        if df_trend_2.empty or not selected_cat_2:
            st.warning("No data available for the selected filters.")
        else:
            total_events_2 = len(df_trend_2)
            cat_events_2 = (df_trend_2["Emergency Category"] == selected_cat_2).sum()
            pct2 = 100 * cat_events_2 / total_events_2 if total_events_2 > 0 else 0.0
            st.markdown(f"**{pct2:.1f}% of events in this period are {selected_cat_2}.**")

            hazard_counts = get_full_bin_data(df_trend_2, selected_cat_2)

            if "# of people assisted" in df_trend_2.columns:
                df_people = df_trend_2[
                    df_trend_2["Emergency Category"] == selected_cat_2
                ].copy()

                assist = (
                    df_people
                    .groupby(['Bin Date', 'Bin Label'])["# of people assisted"]
                    .sum()
                    .reset_index()
                    .rename(columns={"# of people assisted": "PeopleAssisted"})
                )
                hazard_data = hazard_counts.merge(
                    assist,
                    on=["Bin Date", "Bin Label"],
                    how="left",
                )
                hazard_data["PeopleAssisted"] = hazard_data["PeopleAssisted"].fillna(0)
            else:
                hazard_data = hazard_counts.copy()
                hazard_data["PeopleAssisted"] = 0

            hazard_data = hazard_data.sort_values("Bin Date").reset_index(drop=True)
            hazard_data["bin_index"] = hazard_data.index

            max_count = hazard_data["Count"].max()
            hazard_data["MaxBin"] = hazard_data["Count"] == max_count

            label_expr_list = [f"'{lbl}'" for lbl in hazard_data["Bin Label"]]
            label_expr = "[" + ",".join(label_expr_list) + "][datum.value]"

            bars = alt.Chart(hazard_data).mark_bar().encode(
                x=alt.X(
                    'bin_index:O',
                    axis=alt.Axis(
                        title=None,
                        labelAngle=90,
                        values=list(range(len(hazard_data))),
                        labelExpr=label_expr,
                    ),
                ),
                y=alt.Y('Count:Q', axis=alt.Axis(title=None)),
                color=alt.condition(
                    "datum.MaxBin",
                    alt.value("#fd8d3c"),
                    alt.value("#4682b4")
                ),
                tooltip=[
                    'Bin Label',
                    alt.Tooltip('Count:Q', title='Events'),
                    alt.Tooltip('PeopleAssisted:Q', title='People Assisted'),
                ],
            )

            trend = alt.Chart(hazard_data).transform_regression(
                'bin_index', 'Count'
            ).mark_line(
                color="orange",
                strokeDash=[8, 4],
                point=False
            ).encode(
                x='bin_index:O',
                y='Count:Q'
            )

            labels = alt.Chart(hazard_data).mark_text(
                dy=-6,
                color='white',
                fontSize=11
            ).encode(
                x='bin_index:O',
                y='Count:Q',
                text=alt.Text('PeopleAssisted:Q', format='.0f'),
            )

            main_chart = (bars + trend + labels).properties(
                height=380,
                width=900,
                title=f"{selected_cat_2} (labels show people assisted)"
            )

            st.altair_chart(main_chart, use_container_width=True)

# =========================
# TAB 3: TOP 10 HAZARDS
# =========================
with tab3:
    st.header("Events Over Time – Top 10 Hazards")

    bin_size_3, start_bin_3, end_bin_3, bins_df_3 = bin_filters("Top_10")

    df_trend_3 = apply_bin_range_filter(bins_df_3, start_bin_3, end_bin_3)

    if df_trend_3.empty:
        st.warning("No data available for the selected filters.")
    else:
        if "Emergency Category" in df_trend_3.columns:
            top10 = (
                df_trend_3["Emergency Category"]
                .value_counts()
                .nlargest(10)
                .index
                .tolist()
            )
        else:
            top10 = []

        if not top10:
            st.warning("No hazards found.")
        else:
            st.subheader("Top 10 Hazard Trends")

            row1 = top10[0:5]
            row2 = top10[5:10]

            total_events_3 = len(df_trend_3)

            def render_row(hazards_in_row):
                if not hazards_in_row:
                    return
                cols = st.columns(len(hazards_in_row))
                for col, hz in zip(cols, hazards_in_row):
                    hz_data = get_full_bin_data(df_trend_3, hz)

                    hz_data = hz_data.sort_values("Bin Date").reset_index(drop=True)
                    hz_data["bin_index"] = hz_data.index

                    max_count = hz_data["Count"].max()
                    hz_data["MaxBin"] = hz_data["Count"] == max_count

                    labels = list(hz_data["Bin Label"])
                    n = len(labels)

                    hz_events = (df_trend_3["Emergency Category"] == hz).sum()
                    pct = 100 * hz_events / total_events_3 if total_events_3 > 0 else 0.0

                    k = 2
                    shown_indices = [i for i in range(n) if i % k == 0]

                    label_strings = []
                    for i, lbl in enumerate(labels):
                        if i in shown_indices:
                            label_strings.append(f"'{lbl}'")
                        else:
                            label_strings.append("''")
                    label_expr = "[" + ",".join(label_strings) + "][datum.value]"

                    x_axis = alt.Axis(
                        title=None,
                        labelAngle=90,
                        values=list(range(n)),
                        labelExpr=label_expr,
                    )

                    bar = alt.Chart(hz_data).mark_bar().encode(
                        x=alt.X('bin_index:O', axis=x_axis),
                        y=alt.Y('Count:Q', axis=alt.Axis(title=None)),
                        color=alt.condition(
                            "datum.MaxBin",
                            alt.value("#fd8d3c"),
                            alt.value("#4682b4")
                        ),
                        tooltip=['Bin Label', 'Count']
                    )

                    line = alt.Chart(hz_data).transform_regression(
                        'bin_index', 'Count'
                    ).mark_line(
                        color="orange",
                        strokeDash=[8, 4],
                        point=False
                    ).encode(
                        x='bin_index:O',
                        y='Count:Q'
                    )

                    title_text = f"{hz}  ({pct:.1f}% of events)"

                    mini_chart = (bar + line).properties(
                        height=220,
                        width=220,
                        title=title_text
                    )

                    with col:
                        st.altair_chart(mini_chart, use_container_width=True)

            render_row(row1)
            render_row(row2)

# =========================
# TAB 4: SEASONALITY
# =========================
with tab4:
    st.header("Seasonality Analysis")

    season_month_options = month_options

    if "Emergency Category" in df.columns:
        all_categories_tab4 = sorted(df["Emergency Category"].dropna().unique())
    else:
        all_categories_tab4 = []

    selected_months_tab4 = st.multiselect(
        "Months",
        season_month_options,
        default=season_month_options
    )

    filtered_season = df.copy()
    if selected_months_tab4:
        filtered_season = filtered_season[
            filtered_season["Month"].isin(selected_months_tab4)
        ]

    if "Emergency Category" not in filtered_season.columns or filtered_season.empty:
        st.warning("No seasonality data available.")
    else:
        cat_counts = (
            filtered_season["Emergency Category"]
            .value_counts()
            .nlargest(12)
        )
        top12_cats = cat_counts.index.tolist()

        filtered_season = filtered_season[
            filtered_season["Emergency Category"].isin(top12_cats)
        ]

        if filtered_season.empty:
            st.warning("No events for the selected months.")
        else:
            ordered_cats = list(cat_counts.index)

            full_index = pd.MultiIndex.from_product(
                [season_month_options, ordered_cats],
                names=["Month", "Emergency Category"]
            )

            season_counts = (
                filtered_season
                .groupby(['Month', 'Emergency Category'])
                .size()
                .reset_index(name='Count')
            )

            season_counts = (
                season_counts
                .set_index(["Month", "Emergency Category"])
                .reindex(full_index, fill_value=0)
                .reset_index()
            )

            total_events = len(filtered_season)

            heatmap_tab, summary_tab = st.tabs(["Heatmap", "Summary"])

            with heatmap_tab:
                heatmap_ecat = alt.Chart(season_counts).mark_rect().encode(
                    x=alt.X('Month:N', sort=season_month_options,
                            axis=alt.Axis(title=None)),
                    y=alt.Y('Emergency Category:N', sort=ordered_cats,
                            axis=alt.Axis(title=None)),
                    color=alt.condition(
                        alt.datum.Count == 0,
                        alt.value('transparent'),
                        alt.Color(
                            'Count:Q',
                            scale=alt.Scale(
                                domain=[0, season_counts['Count'].max()],
                                range=['#008000', '#ff0000']
                            )
                        )
                    ),
                    opacity=alt.condition(
                        alt.datum.Count == 0,
                        alt.value(0),
                        alt.value(1)
                    ),
                    tooltip=['Month', 'Emergency Category', 'Count']
                ).properties(
                    width=420,
                    height=340,
                    title="Top 12 Emergency Categories – Seasonality"
                )

                st.altair_chart(heatmap_ecat, use_container_width=True)

            with summary_tab:
                st.markdown("### Peak Month by Top 12 Hazards")

                cat_totals = (
                    season_counts
                    .groupby("Emergency Category")["Count"]
                    .sum()
                    .reset_index()
                )
                cat_totals = cat_totals.sort_values("Count", ascending=False).head(12)

                cards_per_row = 6
                hazards = cat_totals["Emergency Category"].tolist()

                for row_start in range(0, len(hazards), cards_per_row):
                    row_hazards = hazards[row_start:row_start + cards_per_row]

                    if row_start >= cards_per_row:
                        st.markdown(
                            "<div style='margin-top:1.5rem;'></div>",
                            unsafe_allow_html=True
                        )

                    cols = st.columns(len(row_hazards))

                    for col, hazard in zip(cols, row_hazards):
                        h_rows = season_counts[
                            season_counts["Emergency Category"] == hazard
                        ]

                        h_sorted = h_rows.sort_values("Count", ascending=False)

                        # top month
                        if not h_sorted.empty and h_sorted.iloc[0]["Count"] > 0:
                            top1_month = h_sorted.iloc[0]["Month"]
                            top1_count = int(h_sorted.iloc[0]["Count"])
                        else:
                            top1_month = "—"
                            top1_count = 0

                        # second month (just name, no percent)
                        if len(h_sorted) > 1 and h_sorted.iloc[1]["Count"] > 0:
                            top2_month = h_sorted.iloc[1]["Month"]
                        else:
                            top2_month = ""

                        # total events for this hazard (denominator)
                        h_total = int(h_sorted["Count"].sum())

                        # PERCENT: share of this hazard's events in the top month
                        pct = 100 * top1_count / h_total if h_total > 0 else 0.0

                        with col:
                            st.markdown(f"**{hazard}**")
                            html = f"""
                            <div style="margin-top:0.25rem;">
                                <div style="font-size:1.1rem; font-weight:700;">
                                    {top1_month}
                                </div>
                                <div style="margin-top:0.25rem; font-size:0.9rem;">
                                    Events in top month: {top1_count}<br>
                                    {pct:.1f}% of this hazard's events
                                </div>
                                <div style="font-size:0.9rem; opacity:0.8;">
                                    {top2_month}
                                </div>
                            </div>
                            """
                            st.markdown(html, unsafe_allow_html=True)
