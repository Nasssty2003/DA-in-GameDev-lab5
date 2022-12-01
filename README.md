# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #5 выполнила:
- Нестерова Анастасия Андреевна
- РИ210947
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | * | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы
Итеграция экономической системы в проект Unity и обучение ML-Agent.

## Задание 1
### Измените параметры файла, yaml-агента и определите какие параметры и как влияю на обучение модели.
Ход работы:

1) Открыть скачанный проект Unity, ознакомиться с его работой.

![Unity](https://user-images.githubusercontent.com/43472988/205011215-aa96cd7e-22a0-4283-bc70-7143240612de.jpg)

2) Поместить файл Economic.yaml в папку с проектом Unity.


3) Перед тем, как перейти к началу обучения, запустите Anaconda Prompt (от имени администратора). Создайте виртуальное пространство следующей команды:


```
conda create -n MLAgent python=3.6.13
```

   Активируйте созданное пространство:
 
```
conda activate MLAgent
```

   Установите набор пакетов из третей лабораторной работы:
   
```
pip install mlagents==0.28.0
pip install torch~=1.7.1 -f
```

   С помощью команды cd перейдите в папку с проектом (где лежит Economic.yaml-файл).
   
   Запустите обучение ML-Агента:
   
   ```
   mlagents-learn Economic.yaml --run-id=Economic –force
   ```
   ![MLA](https://user-images.githubusercontent.com/43472988/205024663-a79818fd-3469-48a1-a202-c5b9c01f1c1e.png)

4) Запустите сцену в Unity. Шарик начинает движение от одного кубика к другому. Для того чтобы ускорить процесс обучения – увеличьте количество префабов TargetAreaEconomiс.

5) Устанавливаем библиотеку TensorBoard (pip install tensorflow) для того, чтобы построить графики оценки результатов обучения.
   Первоначальные данные:

![1](https://user-images.githubusercontent.com/43472988/205028611-5d11ffc0-e910-454e-b74e-a4bf109573bf.png)

![2](https://user-images.githubusercontent.com/43472988/205028628-2ed25c0b-40cb-47bb-9506-587e47d5764d.png)

![3](https://user-images.githubusercontent.com/43472988/205028302-009410e4-8485-4d29-9aa3-2179da699a9b.png)

6) Меняем параметры yaml-агента:

- batch_size (с 1024 на 4096)

```
behaviors:
  Economic:
    trainer_type: ppo
    hyperparameters:
      batch_size: 4096
      buffer_size: 10240
      learning_rate: 3.0e-4
      learning_rate_schedule: linear
      beta: 1.0e-2
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3      
    network_settings:
      normalize: false
      hidden_units: 128
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    checkpoint_interval: 500000
    max_steps: 750000
    time_horizon: 64
    summary_freq: 5000
    self_play:
      save_steps: 20000
      team_change: 100000
      swap_steps: 10000
      play_against_latest_model_ratio: 0.5
      window: 10
```

Полученные графики:

![1ю1](https://user-images.githubusercontent.com/43472988/205039826-06806beb-7bd3-4bbd-93a6-526f8d519b87.png)

![1ю2](https://user-images.githubusercontent.com/43472988/205039831-71f2494a-a08f-4aad-8847-90a83fba0f3d.png)


-buffer_size (с 10240 на 150)

```
behaviors:
  Economic:
    trainer_type: ppo
    hyperparameters:
      batch_size: 1024
      buffer_size: 150
      learning_rate: 3.0e-4
      learning_rate_schedule: linear
      beta: 1.0e-2
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3      
    network_settings:
      normalize: false
      hidden_units: 128
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    checkpoint_interval: 500000
    max_steps: 750000
    time_horizon: 64
    summary_freq: 5000
    self_play:
      save_steps: 20000
      team_change: 100000
      swap_steps: 10000
      play_against_latest_model_ratio: 0.5
      window: 10
```

Полученные графики:

![2ю1](https://user-images.githubusercontent.com/43472988/205084197-d5b059d1-46be-4a6d-9619-b6bdc80d6338.png)

![2ю2](https://user-images.githubusercontent.com/43472988/205084200-82683c4d-8034-4bab-bcb4-f4a3c58d011e.png)

## Задание 2

## Построить графики зависимости количества эпох от ошибки обучения. Указать от чего зависит необходимое количество эпох обучения.

Для создания графиков я собрала данные благодаря проеку в Unity. Меняя количество эпох, я получала различные значения для каждой логической операции , которые и записывала в таблицу, после чего создала диаграмму по этой таблице.

![Снимок экрана 2022-11-29 в 02 04 19](https://user-images.githubusercontent.com/43472988/204380921-ee47f941-b923-4d37-9d7d-4b8df65c63f3.png)

Рассмотрев таюлицу, её данные и графики, а также изучив лекцию, я сделала вывод, что необходимое количество эпох обучения зависит от значений bias(смещение) и weights(вес). Нижеприведённая часть данного код работы перцептрона является подтверждением данной гипотезы, а в частности методы DotProductBias и CalcOutput:

```
double DotProductBias(double[] v1, double[] v2) 
	{
		if (v1 == null || v2 == null)
			return -1;
	 
		if (v1.Length != v2.Length)
			return -1;
	 
		double d = 0;
		for (int x = 0; x < v1.Length; x++)
		{
			d += v1[x] * v2[x];
		}

		d += bias;
	 
		return d;
	}

	double CalcOutput(int i)
	{
		double dp = DotProductBias(weights,ts[i].input);
		if(dp > 0) return(1);
		return (0);
	}
```
	
## Выводы
Лабораторная работа познакомила меня с такой моделью, как перцептрон. Я смогла понаблюдать за его принципами и работой, изучила код на python. Также я поработала с различными логическими операциями и визуализировала их работу в проекте Unity. 

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**

