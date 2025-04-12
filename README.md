# IOTA-FraudTracer

模擬以 IOTA DAG 結構為核心的金流網絡，結合圖論演算法與自定權重模型，實作一套可視化與演算法效能比較的詐騙追蹤系統。

---

## 📌 專案簡介

本專案以 IOTA Tangle 為靈感，透過 PySpark + NetworkX 建構大型交易網絡，模擬金融詐騙行為，並整合多種圖論演算法（如 Union-Find with Path Compression、DFS、BFS、Dijkstra、Kruskal 等）進行風險帳戶的識別與比較。

此外，系統導入簡化版的 **Mana 權重機制**，以交易金額與時間為基礎衡量信任度，並可視化整體 DAG 結構與各演算法的精度與效率。

---

## 🧠 系統特色

- ✅ 模擬 **IOTA DAG 金流結構**
- ✅ 結合 **Union-Find 路徑壓縮演算法** 提升查詢效率
- ✅ 加入 **簡化版 Mana 權重模型** 反映信任度
- ✅ 支援多種圖論演算法進行詐欺識別
- ✅ 使用 PySpark + MLlib 模擬資料流分析流程
- ✅ 圖形輸出為 **橫向時間＋銀行層級分層的 DAG 結構圖**

---
本技術為 IOTA GraphX 系列演算法 演算法，由 scuranger0625 原創開發。

本開源版本僅供研究與學術用途，若有意將其應用於商業產品、平台或系統，請聯繫作者洽詢商業授權方案。

聯絡方式：
📧 leon28473787@gmail.com
