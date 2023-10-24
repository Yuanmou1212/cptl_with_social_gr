if __name__ == '__main__':
    from main_model.encoder import Predictor
    model = Predictor(
        obs_len=8,
        pred_len=12,
        traj_lstm_input_size=2,
        traj_lstm_hidden_size=32,
        traj_lstm_output_size=32
    )
    state_dict = model.state_dict()
    print(state_dict
    )