import pandas as pd

df = pd.read_excel("Attachment/Attachment1.xlsx", sheet_name="Sheet3")

check = (
    df.groupby("推广单元ID")["方案ID"]
      .nunique()
      .sort_values(ascending=False)
)



print(check)

mapping = (
    df[["方案ID", "推广单元ID"]]
    .drop_duplicates()
    .sort_values(["方案ID", "推广单元ID"])
)

print(mapping)


print(
    df[df["推广单元ID"] == 9811363528][
        ["方案ID", "推广单元ID", "关键词"]
    ].head(20)
)

keyword_unit_count = (
    df.groupby("关键词")["推广单元ID"]
      .nunique()
)

cross_unit = keyword_unit_count[keyword_unit_count > 1]

print("跨推广单元的关键词数量：", len(cross_unit))
print(cross_unit.head(30))

print('-'*30)

unit = df[df["推广单元ID"] == 9811363528]

print("这个推广单元的记录数：", len(unit))
print("这个推广单元的不同关键词数：", unit["关键词"].nunique())

print("Sheet3 总行数：", len(df))
print("不同关键词数量：", df["关键词"].nunique())