python train.py --env_id="TestTask-v1" --evaluate --checkpoint=/home/zhw/UCSD/RobotLearning-CSE276F/proj/runs/TestTask-v1__train__1__1749035823/final_ckpt.pt --num_eval_envs=1 --num_eval_steps=1000





python train.py --env_id="TestTask-v1" --num_envs=2048 --update_epochs=8 --num_minibatches=32 --total_timesteps=4_000_000 --eval_freq=10 --num-steps=20 --num_eval-steps=100
