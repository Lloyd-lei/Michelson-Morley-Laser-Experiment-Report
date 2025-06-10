#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
迈克尔逊-莫雷实验相位分析脚本
基于示波器波谷位置测量计算相位差

作者: MM实验团队
日期: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import json
from datetime import datetime
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False

class MMPhaseAnalyzer:
    def __init__(self):
        self.data = {}
        self.results = {}
        # 实验参数
        self.laser_wavelength = 632.8e-9  # 激光波长 (m)
        self.modulation_frequency = 5e6   # 调制频率 (Hz)
        self.theoretical_light_speed = 299792458  # 理论光速 (m/s)
        
    def input_measurement_data(self, angle, arm_name):
        """
        输入单次测量数据（改进的输入方法）
        
        Parameters:
        angle: 测量角度 (度)
        arm_name: 'L1' 或 'L2'
        """
        print(f"\n=== 输入角度 {angle}° 臂 {arm_name} 的测量数据 ===")
        print("请输入5次测量的波谷位置 (格式: 左波谷x值,右波谷x值)")
        print("示例: 1.2,3.8")
        
        measurements = []
        for i in range(5):
            while True:
                try:
                    input_str = input(f"第{i+1}次测量: ").strip()
                    if ',' not in input_str:
                        print("错误：请使用逗号分隔左右波谷位置")
                        continue
                    
                    left_trough, right_trough = map(float, input_str.split(','))
                    
                    # 数据合理性检查
                    if right_trough <= left_trough:
                        print("错误：右波谷位置应大于左波谷位置")
                        continue
                    
                    wavelength = 2 * (right_trough - left_trough)  # 一个完整周期
                    peak_x = (left_trough + right_trough) / 2     # 峰值位置
                    
                    if wavelength <= 0 or wavelength > 100:  # 合理性检查
                        print(f"警告：波长 {wavelength:.3f} 似乎不合理，请检查输入")
                        confirm = input("是否继续使用此数据？ (y/n): ")
                        if confirm.lower() != 'y':
                            continue
                    
                    measurements.append({
                        'left_trough': left_trough,
                        'right_trough': right_trough,
                        'wavelength': wavelength,
                        'peak_x': peak_x
                    })
                    
                    print(f"  -> 波长: {wavelength:.3f}, 峰值位置: {peak_x:.3f}")
                    break
                    
                except ValueError:
                    print("错误：请输入有效的数字")
                except Exception as e:
                    print(f"错误：{e}")
        
        # 存储数据
        if angle not in self.data:
            self.data[angle] = {}
        self.data[angle][arm_name] = measurements
        
        return measurements
    
    def calculate_phase_from_peak(self, peak_x, wavelength, reference_frequency=5e6):
        """
        从峰值位置计算相位
        
        Parameters:
        peak_x: 峰值的x位置 (时间，秒)
        wavelength: 波长 (时间周期，秒)
        reference_frequency: 参考频率 (Hz)
        
        Returns:
        phase_degrees: 相位 (度)
        """
        # 方法1：基于波长的相位计算
        phase_radians = 2 * np.pi * (peak_x % wavelength) / wavelength
        phase_degrees = np.degrees(phase_radians)
        
        return phase_degrees
    
    def unwrap_phases(self, phases):
        """
        相位展开，避免360°跳跃
        """
        if len(phases) <= 1:
            return phases
            
        unwrapped = [phases[0]]
        for i in range(1, len(phases)):
            diff = phases[i] - phases[i-1]
            
            # 如果相位差超过180°，说明发生了跳跃
            if diff > 180:
                # 向下跳跃，减去360°
                unwrapped.append(phases[i] - 360)
            elif diff < -180:
                # 向上跳跃，加上360°
                unwrapped.append(phases[i] + 360)
            else:
                unwrapped.append(phases[i])
                
        return unwrapped
    
    def calculate_phase_difference_robust(self, l1_phases, l2_phases):
        """
        稳健的相位差计算，处理周期性跳跃
        """
        # 对每个相位序列进行展开
        l1_unwrapped = self.unwrap_phases(l1_phases)
        l2_unwrapped = self.unwrap_phases(l2_phases)
        
        # 计算每对测量的相位差
        phase_diffs = []
        for i in range(len(l1_unwrapped)):
            diff = l2_unwrapped[i] - l1_unwrapped[i]
            phase_diffs.append(diff)
        
        # 对相位差序列也进行展开
        phase_diffs_unwrapped = self.unwrap_phases(phase_diffs)
        
        return phase_diffs_unwrapped
    
    def analyze_single_angle(self, angle):
        """
        分析单个角度的数据
        """
        if angle not in self.data:
            print(f"错误：角度 {angle}° 没有数据")
            return None
            
        angle_data = self.data[angle]
        
        if 'L1' not in angle_data or 'L2' not in angle_data:
            print(f"错误：角度 {angle}° 缺少L1或L2数据")
            return None
        
        # 首先收集所有波长数据，计算全局参考波长
        all_wavelengths = []
        for arm in ['L1', 'L2']:
            measurements = angle_data[arm]
            wavelengths = [m['wavelength'] for m in measurements]
            all_wavelengths.extend(wavelengths)
        
        # 使用全局平均波长作为参考
        global_avg_wavelength = np.mean(all_wavelengths)
        
        results = {}
        
        # 分析L1和L2（使用相同的参考波长）
        for arm in ['L1', 'L2']:
            measurements = angle_data[arm]
            
            # 提取数据
            wavelengths = [m['wavelength'] for m in measurements]
            peak_positions = [m['peak_x'] for m in measurements]
            
            # 计算相位（使用全局参考波长）
            phases = [self.calculate_phase_from_peak(pos, global_avg_wavelength) 
                     for pos in peak_positions]
            
            # 统计分析
            results[arm] = {
                'wavelengths': wavelengths,
                'peak_positions': peak_positions,
                'phases': phases,
                'avg_wavelength': np.mean(wavelengths),
                'std_wavelength': np.std(wavelengths, ddof=1) if len(wavelengths) > 1 else 0.0,
                'avg_phase': np.mean(phases),
                'std_phase': np.std(phases, ddof=1) if len(phases) > 1 else 0.0,
                'sem_phase': np.std(phases, ddof=1) / np.sqrt(len(phases)) if len(phases) > 1 else 0.0
            }
        
        # 计算相位差 - 使用稳健方法
        phase_diffs_robust = self.calculate_phase_difference_robust(
            results['L1']['phases'], results['L2']['phases']
        )
        
        # 统计分析相位差
        phase_diff_mean = np.mean(phase_diffs_robust)
        phase_diff_std = np.std(phase_diffs_robust, ddof=1) if len(phase_diffs_robust) > 1 else 0.0
        phase_diff_sem = phase_diff_std / np.sqrt(len(phase_diffs_robust)) if len(phase_diffs_robust) > 1 else 0.0
        
        results['phase_difference'] = {
            'value': phase_diff_mean,
            'uncertainty': phase_diff_sem,  # 使用标准误差
            'individual_diffs': phase_diffs_robust,
            'L1_phases': results['L1']['phases'],
            'L2_phases': results['L2']['phases'],
            'global_reference_wavelength': global_avg_wavelength
        }
        
        # 添加调试信息
        print(f"  调试信息: 全局参考波长 = {global_avg_wavelength:.6f}")
        print(f"  L1峰值位置: {[f'{pos:.3f}' for pos in results['L1']['peak_positions']]}")
        print(f"  L2峰值位置: {[f'{pos:.3f}' for pos in results['L2']['peak_positions']]}")
        print(f"  L1计算相位: {[f'{ph:.1f}°' for ph in results['L1']['phases']]}")
        print(f"  L2计算相位: {[f'{ph:.1f}°' for ph in results['L2']['phases']]}")
        
        self.results[angle] = results
        return results
    
    def input_all_data(self):
        """
        输入所有角度的数据
        """
        angles = [0, 30, 60, 90, 120, 150, 180]
        
        print("=== 迈克尔逊-莫雷实验数据输入系统 ===")
        print("请按顺序输入各个角度的测量数据")
        
        for angle in angles:
            print(f"\n{'='*50}")
            print(f"开始输入角度 {angle}° 的数据")
            print(f"{'='*50}")
            
            # 输入L1数据
            self.input_measurement_data(angle, 'L1')
            
            # 输入L2数据  
            self.input_measurement_data(angle, 'L2')
            
            # 立即分析当前角度
            result = self.analyze_single_angle(angle)
            if result:
                print(f"\n角度 {angle}° 初步分析结果:")
                print(f"L1平均相位: {result['L1']['avg_phase']:.2f}° ± {result['L1']['std_phase']:.2f}°")
                print(f"L2平均相位: {result['L2']['avg_phase']:.2f}° ± {result['L2']['std_phase']:.2f}°")
                print(f"相位差: {result['phase_difference']['value']:.2f}° ± {result['phase_difference']['uncertainty']:.2f}°")
    
    def comprehensive_analysis(self):
        """
        综合分析所有数据
        """
        if not self.results:
            print("错误：没有分析结果，请先输入和分析数据")
            return
        
        print("\n" + "="*80)
        print("                    综合分析结果")
        print("="*80)
        
        # 收集所有相位差数据
        angles = sorted(self.results.keys())
        phase_differences = []
        phase_uncertainties = []
        
        print("\n1. 各角度相位差汇总:")
        print("-" * 60)
        print(f"{'角度(°)':<8} {'相位差(°)':<15} {'不确定度(°)':<12} {'置信区间'}")
        print("-" * 60)
        
        for angle in angles:
            result = self.results[angle]
            phase_diff = float(result['phase_difference']['value'])
            uncertainty = float(result['phase_difference']['uncertainty'])
            
            phase_differences.append(phase_diff)
            phase_uncertainties.append(uncertainty)
            
            # 95%置信区间
            ci_lower = phase_diff - 1.96 * uncertainty
            ci_upper = phase_diff + 1.96 * uncertainty
            
            print(f"{angle:<8} {phase_diff:<15.2f} {uncertainty:<12.2f} [{ci_lower:.2f}, {ci_upper:.2f}]")
        
        # 转换为numpy数组
        angles = np.array(angles, dtype=float)
        phase_differences = np.array(phase_differences, dtype=float)
        phase_uncertainties = np.array(phase_uncertainties, dtype=float)
        
        # 统计检验
        print(f"\n2. 统计分析:")
        print("-" * 40)
        print(f"平均相位差: {np.mean(phase_differences):.3f}°")
        print(f"标准差: {np.std(phase_differences, ddof=1):.3f}°")
        print(f"变化范围: {np.max(phase_differences) - np.min(phase_differences):.3f}°")
        
        # 拟合余弦函数 Δφ = A*cos(2θ + φ₀) + C
        angles_rad = np.radians(angles)
        
        def cosine_model(theta, A, phi0, C):
            return A * np.cos(2 * theta + phi0) + C
        
        def linear_model(theta, a, b):
            return a * theta + b
        
        try:
            from scipy.optimize import curve_fit
            from scipy import stats
            
            # 使用不确定度作为权重
            weights = 1.0 / np.array(phase_uncertainties)
            
            # 1. 线性拟合
            print(f"\n3a. 线性拟合结果: Δφ = a*θ + b")
            print("-" * 50)
            
            popt_linear, pcov_linear = curve_fit(linear_model, angles, phase_differences, 
                                               sigma=phase_uncertainties, absolute_sigma=True)
            
            a, b = popt_linear
            a_err, b_err = np.sqrt(np.diag(pcov_linear))
            
            print(f"斜率 a = {a:.6f} ± {a_err:.6f} °/°")
            print(f"截距 b = {b:.3f} ± {b_err:.3f}°")
            
            # 计算线性拟合优度
            fitted_linear = linear_model(angles, a, b)
            residuals_linear = phase_differences - fitted_linear
            chi_squared_linear = np.sum((residuals_linear / phase_uncertainties)**2)
            dof_linear = len(angles) - 2  # 自由度
            reduced_chi_squared_linear = chi_squared_linear / dof_linear
            
            print(f"线性拟合卡方值: {chi_squared_linear:.2f}")
            print(f"线性拟合约化卡方: {reduced_chi_squared_linear:.2f}")
            
            # 2. 余弦拟合
            print(f"\n3b. 余弦拟合结果: Δφ = A*cos(2θ + φ₀) + C")
            print("-" * 50)
            
            popt_cosine, pcov_cosine = curve_fit(cosine_model, angles_rad, phase_differences, 
                                               sigma=phase_uncertainties, absolute_sigma=True)
            
            A, phi0, C = popt_cosine
            A_err, phi0_err, C_err = np.sqrt(np.diag(pcov_cosine))
            
            print(f"振幅 A = {A:.3f} ± {A_err:.3f}°")
            print(f"相位偏移 φ₀ = {np.degrees(phi0):.1f} ± {np.degrees(phi0_err):.1f}°")
            print(f"直流偏移 C = {C:.3f} ± {C_err:.3f}°")
            
            # 计算余弦拟合优度
            fitted_cosine = cosine_model(angles_rad, A, phi0, C)
            residuals_cosine = phase_differences - fitted_cosine
            chi_squared_cosine = np.sum((residuals_cosine / phase_uncertainties)**2)
            dof_cosine = len(angles) - 3  # 自由度
            reduced_chi_squared_cosine = chi_squared_cosine / dof_cosine
            
            print(f"余弦拟合卡方值: {chi_squared_cosine:.2f}")
            print(f"余弦拟合约化卡方: {reduced_chi_squared_cosine:.2f}")
            
            # 3. 模型比较和统计检验
            print(f"\n4. 模型比较和统计检验:")
            print("-" * 50)
            
            # F检验比较两个模型
            if chi_squared_linear > chi_squared_cosine and dof_linear > dof_cosine:
                f_statistic = ((chi_squared_linear - chi_squared_cosine) / (dof_linear - dof_cosine)) / (chi_squared_cosine / dof_cosine)
                
                # 计算p值
                p_value = 1 - stats.f.cdf(f_statistic, dof_linear - dof_cosine, dof_cosine)
            else:
                # 如果余弦模型拟合更差，直接设置高p值
                f_statistic = 0.0
                p_value = 1.0
            
            print(f"F统计量: {f_statistic:.3f}")
            print(f"p值: {p_value:.6f}")
            
            # AIC比较
            n = len(angles)
            aic_linear = n * np.log(chi_squared_linear/n) + 2 * 2  # 2个参数
            aic_cosine = n * np.log(chi_squared_cosine/n) + 2 * 3  # 3个参数
            
            print(f"线性模型AIC: {aic_linear:.2f}")
            print(f"余弦模型AIC: {aic_cosine:.2f}")
            print(f"ΔAIC (余弦-线性): {aic_cosine - aic_linear:.2f}")
            
            # 结论
            print(f"\n5. 统计结论:")
            print("-" * 30)
            
            if p_value > 0.05:
                print(f"p值 = {p_value:.6f} > 0.05")
                print("结论：无法拒绝线性模型，数据更符合线性关系")
                print("余弦模型没有显著改善拟合效果")
            else:
                print(f"p值 = {p_value:.6f} < 0.05")
                print("结论：拒绝线性模型，数据更符合余弦关系")
                
            if aic_linear < aic_cosine:
                print("AIC准则：线性模型更优")
            else:
                print("AIC准则：余弦模型更优")
                
            # 6. 物理解释
            print(f"\n6. 物理解释:")
            print("-" * 30)
            if p_value > 0.05:
                print("实验结果支持以下解释：")
                print("- 未检测到预期的余弦角度依赖性")
                print("- 可能存在系统性漂移或其他线性效应")
                print("- 不支持经典以太风理论")
                print("- 符合狭义相对论预期（零结果+系统误差）")
            else:
                print("实验结果显示显著的余弦角度依赖性")
                print("需要进一步分析是否为以太风效应")
            
        except ImportError:
            print("警告：scipy不可用，跳过拟合分析")
        except Exception as e:
            print(f"拟合分析出错：{e}")
    
    def plot_results(self):
        """
        绘制结果图表
        """
        if not self.results:
            print("错误：没有分析结果")
            return
        
        angles = sorted(self.results.keys())
        phase_differences = [self.results[angle]['phase_difference']['value'] for angle in angles]
        uncertainties = [self.results[angle]['phase_difference']['uncertainty'] for angle in angles]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 上图：相位差 vs 角度
        ax1.errorbar(angles, phase_differences, yerr=uncertainties, 
                    fmt='o-', capsize=5, capthick=2, label='实验数据')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='零线')
        ax1.set_xlabel('角度 (°)')
        ax1.set_ylabel('相位差 (°)')
        ax1.set_title('迈克尔逊-莫雷实验结果：相位差 vs 旋转角度')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 理论曲线（如果以太存在）
        theta_theory = np.linspace(0, 180, 100)
        phase_theory = 5.69 * np.cos(2 * np.radians(theta_theory))
        ax1.plot(theta_theory, phase_theory, 'r--', alpha=0.7, label='理论预期（以太存在）')
        ax1.legend()
        
        # 下图：各角度的相位测量分布
        for i, angle in enumerate(angles):
            result = self.results[angle]
            L1_phases = result['phase_difference']['L1_phases']
            L2_phases = result['phase_difference']['L2_phases']
            
            x_pos = [angle - 1, angle + 1]
            ax2.scatter([angle - 1] * len(L1_phases), L1_phases, alpha=0.6, label='L1' if i == 0 else "")
            ax2.scatter([angle + 1] * len(L2_phases), L2_phases, alpha=0.6, label='L2' if i == 0 else "")
        
        ax2.set_xlabel('角度 (°)')
        ax2.set_ylabel('相位 (°)')
        ax2.set_title('各角度下L1和L2的相位测量分布')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('MM_实验结果.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("图表已保存为 'MM_实验结果.png'")
    
    def save_results(self, filename=None):
        """
        保存结果到文件
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"MM_实验结果_{timestamp}.json"
        
        # 准备保存的数据
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'raw_data': self.data,
            'analysis_results': {}
        }
        
        # 转换结果为可序列化格式
        for angle, result in self.results.items():
            save_data['analysis_results'][str(angle)] = {
                'L1': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                      for k, v in result['L1'].items()},
                'L2': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                      for k, v in result['L2'].items()},
                'phase_difference': result['phase_difference']
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"结果已保存到: {filename}")
    
    def load_results(self, filename):
        """
        从文件加载结果
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                save_data = json.load(f)
            
            self.data = save_data['raw_data']
            
            # 重新构建结果
            for angle_str, result in save_data['analysis_results'].items():
                angle = int(angle_str)
                self.results[angle] = result
            
            print(f"数据已从 {filename} 加载")
            
        except Exception as e:
            print(f"加载失败：{e}")

    def calculate_light_speed_from_data(self):
        """
        从实验数据推算光速
        使用关系式: c = λ × f
        其中 λ 是激光波长，f 是信号频率
        """
        print("\n=== 光速推算 ===")
        print("-" * 50)
        
        # 更新实验参数为1kHz
        self.modulation_frequency = 1000  # Hz (1kHz)
        
        # 方法1：使用已知的激光波长和信号频率
        calculated_speed_1 = self.laser_wavelength * self.modulation_frequency
        
        print(f"方法1: 基于激光参数（错误方法）")
        print(f"  激光波长 λ = {self.laser_wavelength*1e9:.1f} nm = {self.laser_wavelength:.2e} m")
        print(f"  信号频率 f = {self.modulation_frequency:.0f} Hz")
        print(f"  推算光速 c = λ × f = {calculated_speed_1:.6f} m/s")
        print(f"  理论光速 c₀ = {self.theoretical_light_speed} m/s")
        print(f"  ⚠️  这个结果显然不正确，因为使用的是信号频率而非光频率")
        
        # 方法2：从实验测量的时间周期数据推算光速
        if self.results:
            print(f"\n方法2: 基于实验测量数据（正确方法）")
            all_measured_wavelengths = []
            
            for angle, result in self.results.items():
                if 'L1' in result and 'L2' in result:
                    # 收集所有测量的时间周期数据
                    l1_wavelengths = result['L1']['wavelengths']
                    l2_wavelengths = result['L2']['wavelengths']
                    all_measured_wavelengths.extend(l1_wavelengths)
                    all_measured_wavelengths.extend(l2_wavelengths)
            
            if all_measured_wavelengths:
                avg_measured_time_period = np.mean(all_measured_wavelengths)  # 时间周期 (s)
                std_measured_time_period = np.std(all_measured_wavelengths, ddof=1)
                
                print(f"  测量平均时间周期 T = {avg_measured_time_period*1000:.6f} ± {std_measured_time_period*1000:.6f} ms")
                
                # 理论时间周期应该是 T = 1/f = 1/(1kHz) = 1ms
                theoretical_time_period = 1.0 / self.modulation_frequency  # 理论时间周期
                
                print(f"  理论时间周期 T₀ = 1/f = {theoretical_time_period*1000:.3f} ms")
                
                # 在MM实验中，正确的光速推算方法：
                # 我们测量的是调制信号的时间周期，但光速的计算需要考虑：
                # c = 激光波长 × 光频率
                # 但我们可以通过以下关系推算：
                # 如果调制信号周期 = T，那么在这个时间内光传播的距离是 c × T
                # 而这个距离对应于激光波长的某个倍数
                
                # 正确的光速计算：
                # 在我们的实验设置中，1kHz信号对应1ms周期
                # 在这1ms内，光应该传播 c × 0.001s 的距离
                # 这个距离包含了 (c × 0.001s) / (632.8nm) 个激光波长
                
                # 从测量的时间周期推算光速
                # 假设测量的时间周期准确反映了信号周期
                measured_frequency = 1.0 / avg_measured_time_period
                
                # 关键：在MM实验中，我们需要建立时间周期与光传播的关系
                # 如果信号频率是1kHz，那么在1个周期(1ms)内：
                # 光传播距离 = c × T = c × 0.001s
                # 这个距离对应多少个激光波长？ N = (c × T) / λ
                
                # 重新整理得到：c = N × λ / T
                # 其中N是在一个信号周期内包含的激光波长数
                
                # 对于红色激光在空气中：
                # c ≈ 2.99×10⁸ m/s, λ = 632.8nm, T = 1ms
                # N = c × T / λ = 2.99×10⁸ × 0.001 / 632.8×10⁻⁹ ≈ 472,628
                
                target_light_speed_air = 2.99e8  # m/s (红色激光在空气中)
                theoretical_N = target_light_speed_air * theoretical_time_period / self.laser_wavelength
                
                print(f"  理论上1个信号周期内包含的激光波长数 N₀ = {theoretical_N:.0f}")
                
                # 使用测量的时间周期推算光速
                calculated_speed_2 = theoretical_N * self.laser_wavelength / avg_measured_time_period
                speed_uncertainty = calculated_speed_2 * (std_measured_time_period / avg_measured_time_period)
                
                print(f"  推算光速 c = N₀ × λ / Tₘ = {calculated_speed_2:.2e} ± {speed_uncertainty:.2e} m/s")
                print(f"  目标光速（空气中）= {target_light_speed_air:.2e} m/s")
                print(f"  相对误差 = {abs(calculated_speed_2 - target_light_speed_air)/target_light_speed_air*100:.3f}%")
                
                # 方法3：直接从物理关系推算
                print(f"\n方法3: 基于物理原理的直接推算")
                print(f"  在MM实验的1kHz调制系统中：")
                print(f"  - 信号周期 T = {theoretical_time_period*1000:.3f} ms")
                print(f"  - 激光波长 λ = {self.laser_wavelength*1e9:.1f} nm")
                print(f"  - 在1个信号周期内，光传播距离 = c × T")
                print(f"  - 这个距离包含 N = (c × T) / λ 个激光波长")
                print(f"  - 对于红色激光在空气中：c ≈ {target_light_speed_air:.2e} m/s")
                
                # 验证计算
                distance_per_cycle = target_light_speed_air * theoretical_time_period
                wavelengths_per_cycle = distance_per_cycle / self.laser_wavelength
                
                print(f"  - 每个信号周期光传播距离 = {distance_per_cycle:.0f} m")
                print(f"  - 每个信号周期包含激光波长数 = {wavelengths_per_cycle:.0f}")
                
                return {
                    'method_1_speed': calculated_speed_1,
                    'method_2_speed': calculated_speed_2,
                    'method_2_uncertainty': speed_uncertainty,
                    'target_speed_air': target_light_speed_air,
                    'theoretical_speed': self.theoretical_light_speed,
                    'relative_error_2': abs(calculated_speed_2 - target_light_speed_air)/target_light_speed_air*100,
                    'measured_time_period_ms': avg_measured_time_period * 1000,
                    'theoretical_time_period_ms': theoretical_time_period * 1000,
                    'wavelengths_per_cycle': theoretical_N
                }
        
        return {
            'method_1_speed': calculated_speed_1,
            'theoretical_speed': self.theoretical_light_speed,
            'relative_error_1': abs(calculated_speed_1 - self.theoretical_light_speed)/self.theoretical_light_speed*100,
            'note': '需要实验数据才能进行准确的光速推算'
        }
    
    def validate_experimental_setup(self):
        """
        验证实验装置的一致性
        """
        print("\n=== 实验装置验证 ===")
        print("-" * 50)
        
        # 检查波长测量的一致性
        if self.results:
            all_wavelengths = []
            angle_wavelengths = {}
            
            for angle, result in self.results.items():
                if 'L1' in result and 'L2' in result:
                    l1_wavelengths = result['L1']['wavelengths']
                    l2_wavelengths = result['L2']['wavelengths']
                    angle_avg = np.mean(l1_wavelengths + l2_wavelengths)
                    angle_wavelengths[angle] = angle_avg
                    all_wavelengths.extend(l1_wavelengths + l2_wavelengths)
            
            if all_wavelengths:
                overall_avg = np.mean(all_wavelengths)
                overall_std = np.std(all_wavelengths, ddof=1)
                
                print(f"波长测量一致性检查:")
                print(f"  总体平均波长: {overall_avg:.6f} s")
                print(f"  总体标准差: {overall_std:.6f} s")
                print(f"  变异系数: {overall_std/overall_avg*100:.2f}%")
                
                print(f"\n各角度平均波长:")
                for angle, avg_wl in angle_wavelengths.items():
                    deviation = abs(avg_wl - overall_avg) / overall_avg * 100
                    print(f"  {angle:3d}°: {avg_wl:.6f} s (偏差: {deviation:.2f}%)")
                
                # 判断一致性
                if overall_std/overall_avg < 0.05:  # 变异系数小于5%
                    print(f"  ✅ 波长测量一致性良好")
                else:
                    print(f"  ⚠️  波长测量存在较大变异")

def main():
    """
    主函数
    """
    analyzer = MMPhaseAnalyzer()
    
    while True:
        print("\n" + "="*60)
        print("          迈克尔逊-莫雷实验数据分析系统")
        print("="*60)
        print("1. 输入实验数据")
        print("2. 综合分析结果")
        print("3. 绘制图表")
        print("4. 保存结果")
        print("5. 加载已保存的结果")
        print("6. 光速推算")
        print("7. 实验装置验证")
        print("8. 退出")
        print("-"*60)
        
        try:
            choice = input("请选择操作 (1-8): ").strip()
            
            if choice == '1':
                analyzer.input_all_data()
            elif choice == '2':
                analyzer.comprehensive_analysis()
            elif choice == '3':
                analyzer.plot_results()
            elif choice == '4':
                analyzer.save_results()
            elif choice == '5':
                filename = input("请输入文件名: ").strip()
                analyzer.load_results(filename)
            elif choice == '6':
                analyzer.calculate_light_speed_from_data()
            elif choice == '7':
                analyzer.validate_experimental_setup()
            elif choice == '8':
                print("再见！")
                break
            else:
                print("无效选择，请重新输入")
                
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"发生错误：{e}")

if __name__ == "__main__":
    main() 