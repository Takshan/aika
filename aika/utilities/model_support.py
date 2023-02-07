from pytorch_lightning.callbacks import Callback


class ScatterPlotCallBack(Callback):
    def __init__(self):
        super().__init__()
        self.state = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused=0
    ):
        self.state.append(outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        # access output using state
        all_outputs = self.state
        print(all_outputs), print(len(all_outputs))


def combine_dict(dct1, dct2) -> dict:
    dct2.update(
        {key: dct1[key] + dct2[key] for key in set(dct1.keys()) & set(dct2.keys())}
    )
    dct2.update({key: dct1[key] for key in set(dct1.keys()) - set(dct2.keys())})
    return dct2


def result_dct(all_dictlist):
    """combine all dict in list

    Args:
        all_dictlist (_type_): list of dict

    Returns:
        _type_: flattened dict
    """
    result = {}
    for tt in all_dictlist:
        try:
            for i, j in tt[0].items():
                # convert all tensor to list if its dict
                if isinstance(j, dict):
                    j = {k: v.tolist() for k, v in j.items()}
                    for l, m in j.items():
                        if l not in result:
                            result[l] = []
                        result[l].append(m)
                    # result = combine_dict(j, result)
                else:
                    if i not in result:
                        result[i] = []
                    result[i].append(j)
            # print(i,j)
        except KeyError as e:
            # print(e)
            for i, j in tt.items():
                if i not in result:
                    result[i] = []
                result[i].append(j)
    return result

