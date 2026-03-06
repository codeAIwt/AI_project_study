from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import random
import pandas as pd
import time

# 初始化浏览器（增强反爬配置）
# 定义初始化浏览器驱动的函数
def init_driver():
    # 创建 Chrome 浏览器选项对象，用于配置浏览器启动参数
    options = webdriver.ChromeOptions()
    # 禁用 Chrome 浏览器的 AutomationControlled 特性，减少被网站识别为自动化脚本的可能性
    options.add_argument('--disable-blink-features=AutomationControlled')
    # 设置请求头中的 User-Agent，模拟真实浏览器访问
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    # 设置浏览器窗口大小为 1920x1080
    options.add_argument('--window-size=1920,1080')
    
    try:
        # 尝试使用 ChromeDriverManager 自动下载并安装 Chrome 浏览器驱动
        service = Service(ChromeDriverManager().install())
    except Exception as e:
        # 若自动安装失败，打印错误信息
        print(f"驱动自动安装失败: {e}")
        # 手动指定 Chrome 浏览器驱动的路径，需要将 '你的chromedriver路径' 替换为实际路径
        service = Service(executable_path='C:\Program Files\Google\Chrome\Application\chrome.exe')
    
    # 使用配置好的选项和服务创建 Chrome 浏览器驱动实例并返回
    return webdriver.Chrome(service=service, options=options)

# 改进的爬取函数（带数量控制）
"""
crawl_weibo_data 函数用于从微博搜索页面爬取指定关键词的微博数据，并根据标签进行分类。
持续爬取直到达到目标数据数量。

:param keyword: 用于搜索微博的关键词
:param label: 数据标签，0 表示谣言，1 表示事实
:param target_count: 需要爬取的目标数据数量
:return: 包含微博内容和对应标签的列表
"""
def crawl_weibo_data(keyword, label, target_count):
    # 调用 init_driver 函数初始化浏览器驱动
    driver = init_driver()
    # 用于存储爬取到的数据
    data_list = []
    # 初始化页码为 1
    page = 1

    try:
        # 当已获取的数据数量小于目标数量时，继续爬取
        while len(data_list) < target_count:
            # 构建微博搜索的 URL，包含关键词和页码
            url = f'https://s.weibo.com/weibo?q= {keyword}&page={page}'
            # 让浏览器访问该 URL
            driver.get(url)

            try:
                # 显式等待 15 秒，直到页面上出现指定的 CSS 选择器对应的元素
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'div.card-wrap'))
                )
            except TimeoutException:
                # 若超时未加载出元素，打印错误信息并跳出循环
                print(f"第{page}页加载超时")
                break

            # 模拟人类滚动行为，随机滚动 2 到 4 次
            for _ in range(random.randint(2,4)):
                # 执行 JavaScript 代码，将页面向下滚动页面高度的 1/3
                driver.execute_script("window.scrollBy(0, document.body.scrollHeight/3)")
                # 随机等待 1 到 2 秒，模拟人类操作速度
                time.sleep(random.uniform(1, 2))

            # 改进的选择器，查找所有 class 为 card-wrap 且不是 card-top 的元素
            cards = driver.find_elements(By.CSS_SELECTOR, 'div.card-wrap:not(.card-top)')
            for card in cards:
                try:
                    # 查找每个卡片中的微博内容，并去除前后空格
                    content = card.find_element(By.CSS_SELECTOR, 'p.txt').text.strip()
                    # 过滤掉长度小于等于 20 且包含“广告”的内容
                    if len(content) > 20 and "广告" not in content:
                        # 将符合条件的内容和标签添加到数据列表中
                        data_list.append([content, label])
                        if len(data_list) >= target_count:
                            # 若已获取的数据数量达到目标数量，跳出循环
                            break
                except NoSuchElementException:
                    # 若未找到指定元素，跳过当前循环
                    continue

            # 打印当前已获取的数据数量和目标数量
            print(f"已获取 {len(data_list)}/{target_count} 条{keyword}数据")
            # 页码加 1，准备爬取下一页
            page += 1
            # 随机等待 3 到 6 秒，避免频繁请求触发反爬机制
            time.sleep(random.uniform(3, 6))

    except Exception as e:
        # 若出现异常，打印异常信息
        print(f"爬取异常: {e}")
    finally:
        # 无论是否出现异常，都关闭浏览器驱动
        driver.quit()

    # 返回数据列表，确保不超过目标数量
    return data_list[:target_count]

if __name__ == "__main__":
    # 配置参数（可根据需要调整）
    TOTAL_DATA = 250
    RUMOR_RATIO = 0.5  # 谣言数据占比
    
    # 计算各类别目标数量
    rumor_target = int(TOTAL_DATA * RUMOR_RATIO)
    fact_target = TOTAL_DATA - rumor_target
    
    # 爬取谣言数据（标签0）
    rumor_data = []
    for kw in ["米哈游谣言","崩坏3谣言","崩坏星穹铁道谣言","原神谣言"]:
        if len(rumor_data) < rumor_target:
            remaining = rumor_target - len(rumor_data)
            rumor_data.extend(crawl_weibo_data(kw, 0, remaining))
    
    # 爬取真实数据（标签1）
    fact_data = []
    for kw in ["崩坏3辟谣","崩坏3官方辟谣","米哈游官方辟谣","米哈游官方公告","崩坏星穹铁道官方公告","原神官方公告"]:
        if len(fact_data) < fact_target:
            remaining = fact_target - len(fact_data)
            fact_data.extend(crawl_weibo_data(kw, 1, remaining))
    
    # 合并数据并打乱顺序
    combined_data = rumor_data + fact_data
    random.shuffle(combined_data)
    
    # 保存数据
    df = pd.DataFrame(combined_data, columns=['内容', '标签'])
    df = df.drop_duplicates(subset=['内容'])
    df = df.head(TOTAL_DATA)  # 最终控制数量
    
    df.to_csv('weibo_honkai_data.csv', index=False, encoding='utf-8-sig')
    print(f"数据采集完成！有效数据量：{len(df)}条（0: {len(df[df['标签']==0])}条，1: {len(df[df['标签']==1])}条）")

# 主要改进点：
# 1. 动态数量控制：持续爬取直到达到目标数量
# 2. 智能分页：自动翻页直到满足数据需求
# 3. 比例配置：通过RUMOR_RATIO参数控制谣言/真实数据比例
# 4. 内容过滤：排除广告和短文本（长度>20）
# 5. 随机化处理：滚动行为、等待时间、结果打乱顺序

# 使用建议：
# 1. 根据网络情况调整等待时间参数
# 2. 可添加代理IP池应对反爬
# 3. 如果遇到验证码，需要手动处理或添加打码平台接口
# 4. 关键词列表可根据具体需求扩展