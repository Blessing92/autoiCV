import csv, io
import logging
import pandas as pd

from django.shortcuts import render
from autoapp.models import Info
from django.http import JsonResponse
from autoapp.forms import UserForm, UserProfileInfoForm
from autoapp.process import extract_dataset_properties

from django.contrib.auth import authenticate, login, logout
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
from django.contrib import messages
from django.contrib.auth.decorators import login_required, permission_required


# Create your views here.
def index(request):
    context_dict = {'text': 'hello world', 'number': 79}
    return render(request, 'autoapp/index.html', context_dict)


@login_required
def special(request):
    return HttpResponse("You are logged in, Nice!")


@login_required
def user_logout(request):
    logout(request)
    return HttpResponseRedirect(reverse('index'))


def other(request):
    return render(request, 'autoapp/upload_csv.html')


def register(request):
    registered = False

    if request.method == "POST":
        user_form = UserForm(data=request.POST)
        profile_form = UserProfileInfoForm(data=request.POST)

        if user_form.is_valid() and profile_form.is_valid():
            user = user_form.save()
            user.set_password(user.password)
            user.save()

            profile = profile_form.save(commit=False)
            profile.user = user

            if 'profile_pic' in request.FILES:
                profile.profile_pic = request.FILES['profile_pic']
            profile.save()

            registered = True
        else:
            print(user_form.errors, profile_form.errors)
    else:
        user_form = UserForm()
        profile_form = UserProfileInfoForm()

    return render(request, 'autoapp/registration.html',
                  {'user_form': user_form,
                   'profile_form': profile_form,
                   'registered': registered})


def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(username=username, password=password)

        if user:
            if user.is_active:
                login(request, user)
                return HttpResponseRedirect(reverse('index'))

            else:
                return HttpResponse("ACCOUNT NOT ACTIVE")
        else:
            print("Someone tried to login and failed!")
            print("Username: {} and password: {}".format(username, password))
            return HttpResponse("Invalid login details supplied!")
    else:
        return render(request, 'autoapp/login.html', {})


def base(request):
    return render(request, 'autoapp/base.html', )


@permission_required('admin.can_add_log_entry')
def contact_upload(request):
    template = "autoapp/contact_upload.html"

    prompt = {
        'order': 'Order of CSV should be first_name, last_name, email, ip_address, message'
    }

    if request.method == 'GET':
        return render(request, template, prompt)

    csv_file = request.FILES['file']
    if not csv_file.name.endswith('.csv'):
        messages.error(request, 'This is not a csv file')

    data_set = csv_file.read().decode('UTF-8')
    io_string = io.StringIO(data_set)
    # skip the header of the csv file
    next(io_string)
    for column in csv.reader(io_string, delimiter=',', quotechar="|"):
        _, created = Info.objects.update_or_create(
            first_name=column[0],
            last_name=column[1],
            email=column[2],
            ip_address=column[3],
            message=column[4]
        )

    context = {}
    return render(request, template, context)


def upload_csv(request):
    data = {}
    properties = {}
    variables = {}
    rows = []

    if "GET" == request.method:
        return render(request, "autoapp/upload_csv.html", data)
    # if not GET, then proceed
    try:
        csv_file = request.FILES["csv_file"]
        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'File is not CSV type')
            return HttpResponseRedirect(reverse("autoapp:upload_csv"))
        # if file is too large, return
        # if csv_file.multiple_chunks():
        #     messages.error(request, "Uploaded file is too big (%.2f MB)." % (csv_file.size / (10000 * 10000),))
        #     return HttpResponseRedirect(reverse("autoapp:upload_csv"))

        file_data = csv_file.read().decode("utf-8")
        io_string = io.StringIO(file_data)
        properties = extract_dataset_properties(io_string)

    except Exception as e:
        logging.getLogger("error_logger").error("Unable to upload file. " + repr(e))
        messages.error(request, "Unable to upload file. " + repr(e))



    #return HttpResponseRedirect(reverse("autoapp:upload_csv"))
    return render(request,"autoapp/upload_csv.html", properties)



def selected_model(request):
    try:
        results = request.GET['model_choice']

    except Exception as e:
        results = "Please select a model"
        # logging.getLogger("error_logger").error("Unable to upload file. " + repr(e))
        # messages.error(request, "Please select a model " + repr(e))

    return render(request,"autoapp/training.html", {'choice': results})


def room(request, room_name):
    return render(request, 'autoapp/chatroom.html', {
        'room_name': room_name
    })
    
