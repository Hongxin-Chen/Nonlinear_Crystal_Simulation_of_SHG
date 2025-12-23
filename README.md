# Nonlinear Crystal SHG Calculator (非线性晶体二次谐波计算器)

## 简介 (Introduction)
这是一个基于 Python 和 Streamlit 开发的光学仿真工具，用于计算常见非线性晶体（如 BBO, LBO, CLBO, KDP 等）在二次谐波产生（SHG）过程中的关键参数。

## 主要功能 (Features)
- **多晶体支持**：内置 BBO, LBO, CLBO, KDP, DKDP 等常用晶体数据库。
- **相位匹配计算**：支持 Type I 和 Type II 相位匹配角的精确计算。
- **有效非线性系数 ($d_{eff}$)**：基于 IEEE 标准和 Kleinman 对称性计算不同切向下的 $d_{eff}$。
- **可视化交互**：提供交互式 Web 界面，实时查看参数变化。
- **可视化图表**：支持 3D 晶体切向示意图和折射率椭球绘制。

**演示图**
<img width="1575" height="551" alt="c58a075d9f84e946291da21ede2ad2a9" src="https://github.com/user-attachments/assets/9301460a-6002-4135-83ce-c7ed678033fb" />
<img width="1136" height="806" alt="72229e5361e2c001dd6982fab563309a" src="https://github.com/user-attachments/assets/5a8c314a-f220-4ebb-9b62-8f72d62df564" />
<img width="1508" height="772" alt="4d6cdd610d6a0779d14132aa54a7d5d5" src="https://github.com/user-attachments/assets/5a721261-3119-4d04-932f-c9eccc8c3d71" />

## 安装与运行 (Installation & Usage)

1. **克隆仓库**
   ```bash
   git clone [https://github.com/Hongxin-Chen/Nonlinear_Crystal_Simulation_of_SHG.git](https://github.com/Hongxin-Chen/Nonlinear_Crystal_Simulation_of_SHG.git)
   cd Nonlinear_Crystal_Simulation_of_SHG
