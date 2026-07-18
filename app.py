import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model and dataframe
with open('pipe.pkl', 'rb') as f:
    pipe = pickle.load(f)
with open('df.pkl', 'rb') as f:
    df = pickle.load(f)

# Static option lists (same as the original Streamlit app)
RAM_OPTIONS = [2, 4, 6, 8, 12, 16, 24, 32, 64]
RESOLUTION_OPTIONS = [
    '1920x1080', '1366x768', '1600x900',
    '3840x2160', '3200x1800', '2880x1800',
    '2560x1600', '2560x1440', '2304x1440'
]
HDD_OPTIONS = [0, 128, 256, 512, 1024, 2048]
SSD_OPTIONS = [0, 8, 128, 256, 512, 1024]


def get_form_options():
    """Build the dropdown choices pulled from the dataframe, same as df['col'].unique()."""
    return {
        'companies': sorted(df['Company'].unique()),
        'types': sorted(df['TypeName'].unique()),
        'cpus': sorted(df['Cpu brand'].unique()),
        'gpus': sorted(df['Gpu brand'].unique()),
        'os_list': sorted(df['os'].unique()),
        'ram_options': RAM_OPTIONS,
        'resolution_options': RESOLUTION_OPTIONS,
        'hdd_options': HDD_OPTIONS,
        'ssd_options': SSD_OPTIONS,
    }


@app.route("/")
def home():
    return render_template("home.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    options = get_form_options()
    prediction = None
    error = None
    form_data = {}

    if request.method == 'POST':
        # Grab raw form values (kept as strings for repopulating the form)
        form_data = request.form.to_dict()

        try:
            company = request.form['company']
            type_ = request.form['type']
            ram = int(request.form['ram'])
            weight = float(request.form['weight'])
            touchscreen = request.form['touchscreen']
            ips = request.form['ips']
            screen_size = float(request.form['screen_size'])
            resolution = request.form['resolution']
            cpu = request.form['cpu']
            hdd = int(request.form['hdd'])
            ssd = int(request.form['ssd'])
            gpu = request.form['gpu']
            os_ = request.form['os']

            # Validate submitted values against known option lists
            # (protects against direct POSTs that bypass the <select> dropdowns)
            if (company not in options['companies']
                    or type_ not in options['types']
                    or cpu not in options['cpus']
                    or gpu not in options['gpus']
                    or os_ not in options['os_list']
                    or resolution not in options['resolution_options']
                    or ram not in options['ram_options']
                    or hdd not in options['hdd_options']
                    or ssd not in options['ssd_options']
                    or touchscreen not in ('Yes', 'No')
                    or ips not in ('Yes', 'No')):
                error = "Please choose valid values from the provided options."
            elif screen_size <= 0:
                error = "Screen size must be greater than 0."
            elif weight <= 0:
                error = "Weight must be greater than 0."
            else:
                touchscreen_val = 1 if touchscreen == "Yes" else 0
                ips_val = 1 if ips == "Yes" else 0

                X_res = int(resolution.split("x")[0])
                Y_res = int(resolution.split("x")[1])
                ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

                query = np.array(
                    [[company, type_, ram, weight, touchscreen_val, ips_val,
                      ppi, cpu, hdd, ssd, gpu, os_]],
                    dtype=object
                )

                predicted_log_price = pipe.predict(query)[0]
                prediction = int(np.exp(predicted_log_price))

        except (KeyError, ValueError, IndexError):
            error = "Please fill in all fields with valid values."
        except Exception:
            # Catch-all so unexpected pipeline/model errors don't crash the app
            error = "Something went wrong while processing your request. Please try again."

    return render_template(
        'index.html',
        options=options,
        prediction=prediction,
        error=error,
        form_data=form_data
    )


if __name__ == '__main__':
    app.run(debug=True)
