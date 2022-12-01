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


- buffer_size (с 10240 на 150)

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

- num_layers (с 2 на 100)

```
behaviors:
  Economic:
    trainer_type: ppo
    hyperparameters:
      batch_size: 1024
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
      num_layers: 100
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

![3ю1](https://user-images.githubusercontent.com/43472988/205085402-34e1a195-275d-49f4-977c-c44cf7258f2f.png)

![3ю2](https://user-images.githubusercontent.com/43472988/205085412-a9105ebb-2a7a-4f99-8dbd-81e8f50f53fe.png)

- epsilon (c 0.2 на 1.2)

```
behaviors:
  Economic:
    trainer_type: ppo
    hyperparameters:
      batch_size: 1024
      buffer_size: 10240
      learning_rate: 3.0e-4
      learning_rate_schedule: linear
      beta: 1.0e-2
      epsilon: 1.2
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

![4ю1](https://user-images.githubusercontent.com/43472988/205099474-e082feb0-0765-43f3-b1c2-94172e92e026.png)

![4ю2](https://user-images.githubusercontent.com/43472988/205099536-f8219229-03ed-4eec-bec8-6f2e4fd75319.png)

- lambd (c 0.95 на 0.1)

```
behaviors:
  Economic:
    trainer_type: ppo
    hyperparameters:
      batch_size: 1024
      buffer_size: 10240
      learning_rate: 3.0e-4
      learning_rate_schedule: linear
      beta: 1.0e-2
      epsilon: 0.2
      lambd: 0.1
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

![5ю1](https://user-images.githubusercontent.com/43472988/205100323-a0d37242-fcfc-4dd1-9578-b52533feff33.png)

![5ю2](https://user-images.githubusercontent.com/43472988/205100330-22d717af-9609-4ec5-ac01-7f08d5d2a51f.png)

Рассмотрев все полученные графики, можно сделать следующие выводы. Изменение многих вышеперечисленных значений не особо влияет на графики, так как они либо не изменяются, либо делюъают это незначительно. Если говорить чуть конкретнее, то график Extrinsic Value Estimate, например, растет вверх при изменении всех значений кроме  epsilon. Также можно заметить, что при изменении параметра num_layers, график меняется относительно первоначальной версии и обучение становится дольше по времени. Но в общем нужно сказать, что данные изменения в большей степени имеют положительный эффект на обучении модели. 

## Задание 2

## Опишите результаты, выведенные в TensorBoard.




	
## Выводы
Лабораторная работа позволила мне больше углубиться в работу с MLAgent и научиться с его помощью внедрять некоторую экономику в проект Unity. Также я меняла параметры yaml-файла и наблюдала за тем, каким образом это влияет на результаты обучения MLAgent. Эти наблюдения, а также анализ, я проводила благодаря библиотеке TensorBoard, которая позволила мне создавать графики, дающие более наглядный формат данных.

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

