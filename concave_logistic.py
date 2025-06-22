import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

# SCAD惩罚函数及梯度计算（基于Lecture_Notes矩阵微积分理论）
def scad(t, alpha=3.7, lambda_=0.1):
    """SCAD凹惩罚函数，满足SCAD''(t) ≤ 0"""
    if np.isscalar(t):
        if t <= lambda_:
            return alpha * t
        elif t <= alpha * lambda_:
            return (-t**2 + 2 * alpha * lambda_ * t - lambda_** 2) / (2 * (alpha - 1))
        else:
            return (alpha + 1) * lambda_** 2 / 2
    else:
        return np.where(
            t <= lambda_,
            alpha * t,
            np.where(
                t <= alpha * lambda_,
                (-t**2 + 2 * alpha * lambda_ * t - lambda_**2) / (2 * (alpha - 1)),
                (alpha + 1) * lambda_** 2 / 2
            )
        )

def scad_gradient(t, alpha=3.7, lambda_=0.1):
    """SCAD惩罚项梯度计算"""
    if np.isscalar(t):
        if t <= lambda_:
            return alpha
        elif t <= alpha * lambda_:
            return (-2 * t + 2 * alpha * lambda_) / (2 * (alpha - 1))
        else:
            return 0
    else:
        return np.where(
            t <= lambda_,
            alpha * np.ones_like(t),
            np.where(
                t <= alpha * lambda_,
                (-2 * t + 2 * alpha * lambda_) / (2 * (alpha - 1)),
                np.zeros_like(t)
            )
        )

# 逻辑回归损失及梯度（基于Lecture_Notes多元线性回归理论）
def log_reg_loss(X, y, w):
    """逻辑回归损失函数 f(x) = (1/n)sum(ln(1+exp(-b_i a_i^T x)))"""
    z = y * (X @ w)
    return np.mean(np.log(1 + np.exp(-z)))

def log_reg_gradient(X, y, w):
    """逻辑回归梯度计算"""
    z = y * (X @ w)
    p = 1 / (1 + np.exp(z))
    return -np.mean(y.reshape(-1, 1) * X * p.reshape(-1, 1), axis=0).flatten()


### 算法实现部分（严格遵循Lecture_Notes优化理论）###

class PGD:
    """近端梯度下降法（PGD），实现Lecture_Notes定理4.1框架"""
    def __init__(self, lambda_=0.1, alpha=3.7, eta=0.001, max_iter=300, tol=1e-4):
        self.lambda_ = lambda_
        self.alpha = alpha
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol
        self.loss_history = []
        self.w_history = []
    
    def fit(self, X, y):
        n, d = X.shape
        w = np.zeros(d)
        self.w_history.append(w.copy())
        self.loss_history.append(self._objective(X, y, w))
        
        for i in range(self.max_iter):
            grad = log_reg_gradient(X, y, w)
            prox_w = self._scad_prox(w - self.eta * grad)
            
            self.w_history.append(prox_w.copy())
            loss = self._objective(X, y, prox_w)
            self.loss_history.append(loss)
            
            if i > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                break
                
            w = prox_w
            
        self.w = w
    
    def _objective(self, X, y, w):
        """目标函数 = 逻辑回归损失 + SCAD惩罚"""
        return log_reg_loss(X, y, w) + self.lambda_ * np.sum(scad(np.abs(w), self.alpha, 1))
    
    def _scad_prox(self, z):
        """SCAD近端算子，实现Lecture_Notes定理6.1"""
        abs_z = np.abs(z)
        prox = np.zeros_like(z)
        lambda_ = self.lambda_
        alpha = self.alpha
        eta = self.eta
        
        mask1 = abs_z <= eta * lambda_
        mask2 = (abs_z > eta * lambda_) & (abs_z <= eta * alpha * lambda_)
        mask3 = abs_z > eta * alpha * lambda_
        
        prox[mask1] = 0
        prox[mask2] = z[mask2] * (abs_z[mask2] - eta * lambda_) / (abs_z[mask2] - eta * lambda_ / (alpha - 1))
        prox[mask3] = z[mask3] * alpha * lambda_ / abs_z[mask3]
        return prox


class APG:
    """加速近端梯度法（APG），基于Lecture_Notes定理5.2加速理论"""
    def __init__(self, lambda_=0.1, alpha=3.7, eta=0.001, max_iter=300, tol=1e-4):
        self.lambda_ = lambda_
        self.alpha = alpha
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol
        self.loss_history = []
        self.w_history = []
        self.lambda_seq = [0]  # 动量参数序列
    
    def fit(self, X, y):
        n, d = X.shape
        w = np.zeros(d)
        y_k = w.copy()
        self.lambda_seq.append(1)  # 初始化动量参数
        
        self.w_history.append(w.copy())
        loss = self._objective(X, y, w)
        self.loss_history.append(loss)
        
        for i in range(self.max_iter):
            lambda_prev = self.lambda_seq[-1]
            lambda_current = (1 + np.sqrt(1 + 4 * lambda_prev** 2)) / 2
            self.lambda_seq.append(lambda_current)
            
            # 构造动量项（Lecture_Notes公式5.12）
            y_k = w + ((lambda_prev - 1) / lambda_current) * (w - self.w_history[-1])
            grad = log_reg_gradient(X, y, y_k)
            prox_w = self._scad_prox(y_k - self.eta * grad)
            
            self.w_history.append(prox_w.copy())
            new_loss = self._objective(X, y, prox_w)
            self.loss_history.append(new_loss)
            
            if i > 0 and abs(new_loss - self.loss_history[-2]) < self.tol:
                break
                
            w = prox_w
            
        self.w = w
    
    def _objective(self, X, y, w):
        return log_reg_loss(X, y, w) + self.lambda_ * np.sum(scad(np.abs(w), self.alpha, 1))
    
    def _scad_prox(self, z):
        abs_z = np.abs(z)
        prox = np.zeros_like(z)
        lambda_ = self.lambda_
        alpha = self.alpha
        eta = self.eta
        
        mask1 = abs_z <= eta * lambda_
        mask2 = (abs_z > eta * lambda_) & (abs_z <= eta * alpha * lambda_)
        mask3 = abs_z > eta * alpha * lambda_
        
        prox[mask1] = 0
        prox[mask2] = z[mask2] * (abs_z[mask2] - eta * lambda_) / (abs_z[mask2] - eta * lambda_ / (alpha - 1))
        prox[mask3] = z[mask3] * alpha * lambda_ / abs_z[mask3]
        return prox


class LLA:
    """线性化Lasso算法（LLA），基于Lecture_Notes第6章线性化理论"""
    def __init__(self, lambda_=0.1, alpha=3.7, eta=0.001, L=10, max_iter=300, tol=1e-4):
        self.lambda_ = lambda_
        self.alpha = alpha
        self.eta = eta
        self.L = L  # 线性化Lipschitz常数
        self.max_iter = max_iter
        self.tol = tol
        self.loss_history = []
        self.w_history = []
    
    def fit(self, X, y):
        n, d = X.shape
        w = np.zeros(d)
        self.w_history.append(w.copy())
        self.loss_history.append(self._objective(X, y, w))
        
        for i in range(self.max_iter):
            grad = log_reg_gradient(X, y, w)
            # 线性化步骤（Lecture_Notes公式6.5）
            z = w - (1/self.L) * grad
            prox_w = self._scad_prox(z)
            
            self.w_history.append(prox_w.copy())
            loss = self._objective(X, y, prox_w)
            self.loss_history.append(loss)
            
            if i > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                break
                
            w = prox_w
            
        self.w = w
    
    def _objective(self, X, y, w):
        return log_reg_loss(X, y, w) + self.lambda_ * np.sum(scad(np.abs(w), self.alpha, 1))
    
    def _scad_prox(self, z):
        abs_z = np.abs(z)
        prox = np.zeros_like(z)
        lambda_ = self.lambda_
        alpha = self.alpha
        eta = self.eta
        
        mask1 = abs_z <= eta * lambda_
        mask2 = (abs_z > eta * lambda_) & (abs_z <= eta * alpha * lambda_)
        mask3 = abs_z > eta * alpha * lambda_
        
        prox[mask1] = 0
        prox[mask2] = z[mask2] * (abs_z[mask2] - eta * lambda_) / (abs_z[mask2] - eta * lambda_ / (alpha - 1))
        prox[mask3] = z[mask3] * alpha * lambda_ / abs_z[mask3]
        return prox


class LQA:
    """Lasso二次近似算法（LQA），实现Lecture_Notes第7章二次近似理论"""
    def __init__(self, lambda_=0.1, alpha=3.7, delta_init=1.0, delta_decay=0.5,
                 max_delta_iter=8, newton_max_iter=5, tol=1e-4):
        self.lambda_ = lambda_
        self.alpha = alpha
        self.delta_init = delta_init
        self.delta_decay = delta_decay
        self.max_delta_iter = max_delta_iter
        self.newton_max_iter = newton_max_iter
        self.tol = tol
        self.loss_history = []
        self.w_history = []
    
    def fit(self, X, y):
        n, d = X.shape
        w = np.zeros(d)
        delta = self.delta_init
        
        for delta_iter in range(self.max_delta_iter):
            for newton_iter in range(self.newton_max_iter):
                grad = self._quadratic_gradient(X, y, w, delta)
                hess = self._quadratic_hessian(X, y, w)
                
                # 牛顿方向计算（Lecture_Notes定理7.2）
                hess = np.diag(np.diag(hess))  # 确保对角Hessian
                direction = -np.linalg.solve(hess + 1e-6 * np.eye(hess.shape[0]), grad)
                w_new = w + direction
                
                self.w_history.append(w_new.copy())
                loss = self._quadratic_objective(X, y, w_new, delta)
                self.loss_history.append(loss)
                
                if np.linalg.norm(w_new - w) < self.tol:
                    break
                    
                w = w_new
            
            # 衰减delta并记录原始目标函数
            delta *= self.delta_decay
            original_loss = self._original_objective(X, y, w)
            self.loss_history.append(original_loss)
            self.w_history.append(w.copy())
            
            if delta < 1e-6:
                break
                
        self.w = w
    
    def _quadratic_objective(self, X, y, w, delta):
        """二次近似目标函数"""
        log_loss = log_reg_loss(X, y, w)
        scad_approx = self.lambda_ * np.sum(self._scad_quadratic(np.abs(w), delta))
        return log_loss + scad_approx
    
    def _scad_quadratic(self, t, delta):
        """SCAD二次近似"""
        return np.where(t <= delta, t**2 / (2 * delta), scad(t, self.alpha, self.lambda_))
    
    def _quadratic_gradient(self, X, y, w, delta):
        """二次近似梯度"""
        log_grad = log_reg_gradient(X, y, w)
        scad_grad = self.lambda_ * self._scad_quadratic_grad(np.abs(w), delta) * np.sign(w)
        return log_grad + scad_grad
    
    def _scad_quadratic_grad(self, t, delta):
        """二次近似梯度计算"""
        return np.where(t <= delta, t / delta, scad_gradient(t, self.alpha, self.lambda_))
    
    def _quadratic_hessian(self, X, y, w):
        """Hessian矩阵计算（对角近似）"""
        z = y * (X @ w)
        p = 1 / (1 + np.exp(z))
        p = p.reshape(-1, 1)  # 规范形状为(样本数, 1)
        
        # 计算对角元素（Lecture_Notes公式7.10）
        hess_diag = np.mean(p * (1 - p) * (X ** 2), axis=0)
        return np.diag(hess_diag)
    
    def _original_objective(self, X, y, w):
        """原始目标函数"""
        return log_reg_loss(X, y, w) + self.lambda_ * np.sum(scad(np.abs(w), self.alpha, 1))


### 实验与可视化（符合期末大作业要求）###

def run_experiment():
    # 数据加载与预处理
    X, y = load_breast_cancer(return_X_y=True)
    y = 2 * y - 1  # 转换为{-1, 1}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 算法实例化（参数调整基于Lecture_Notes理论）
    lambda_ = 0.1
    alpha = 3.7
    algorithms = {
        'PGD': PGD(lambda_=lambda_, alpha=alpha, eta=0.001, max_iter=300),
        'APG': APG(lambda_=lambda_, alpha=alpha, eta=0.001, max_iter=300),
        'LLA': LLA(lambda_=lambda_, alpha=alpha, eta=0.001, max_iter=300),
        'LQA': LQA(lambda_=lambda_, alpha=alpha, max_delta_iter=8)
    }
    
    # 训练算法并记录时间
    results = {}
    time_results = {}
    for name, algo in algorithms.items():
        print(f"Executing {name} algorithm...")
        start_time = time.time()
        algo.fit(X_train, y_train)
        end_time = time.time()
        results[name] = algo
        time_results[name] = end_time - start_time
        print(f"{name} execution time: {end_time - start_time:.4f} seconds")
    
    # 绘制PGD损失可视化（单独图表）
    plt.figure(figsize=(10, 6))
    plt.plot(results['PGD'].loss_history, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.title('PGD Convergence Curve for SCAD-Regularized Logistic Regression')
    plt.grid(True, alpha=0.3)
    plt.savefig('pgd_loss_visualization.png', dpi=300)
    plt.close()
    
    # 多算法收敛曲线对比
    plt.figure(figsize=(12, 8))
    for name, algo in results.items():
        plt.plot(algo.loss_history[:min(200, len(algo.loss_history))], label=name)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.title('Convergence Comparison of Optimization Algorithms')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('algorithm_comparison.png', dpi=300)
    plt.close()
    
    # 性能评估（准确率、稀疏度）
    eval_results = {}
    for name, algo in results.items():
        w = algo.w
        # 测试集准确率
        z = X_test @ w
        y_pred = np.sign(z)
        accuracy = np.mean(y_pred == y_test)
        # 解的稀疏度
        sparsity = np.sum(w == 0) / len(w)
        eval_results[name] = {
            'accuracy': accuracy,
            'sparsity': sparsity,
            'time': time_results[name]
        }
        print(f"{name}: Accuracy={accuracy:.4f}, Sparsity={sparsity:.4f}, Time={time_results[name]:.4f}s")
    
    return results, eval_results


if __name__ == "__main__":
    results, eval_results = run_experiment()